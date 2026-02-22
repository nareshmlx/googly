import hmac
from typing import Annotated, Any

import jwt
import structlog
from fastapi import Depends, Header, HTTPException
from jwt import PyJWKClient

from app.core.config import settings

logger = structlog.get_logger(__name__)

# Eagerly initialise the JWKS client so there is no lazy-init race condition.
# PyJWKClient.__init__ is pure CPU (no network I/O) — network happens on first
# get_signing_key_from_jwt() call, which is protected by PyJWKClient's own
# internal cache and thread-safe key-fetch logic.
_jwks_client: PyJWKClient | None = (
    PyJWKClient(settings.CLERK_JWKS_URL, cache_keys=True) if settings.CLERK_JWKS_URL else None
)


def _validate_optional_clerk_claims(payload: dict[str, Any]) -> None:
    """Validate optional Clerk issuer/audience claims when configured."""
    expected_issuer = settings.CLERK_EXPECTED_ISSUER
    if expected_issuer and payload.get("iss") != expected_issuer:
        raise HTTPException(status_code=401, detail="Invalid token: invalid issuer")

    expected_audience = settings.CLERK_EXPECTED_AUDIENCE
    if not expected_audience:
        return

    token_audience = payload.get("aud")
    if isinstance(token_audience, list):
        valid_audience = expected_audience in token_audience
    else:
        valid_audience = token_audience == expected_audience

    if not valid_audience:
        raise HTTPException(status_code=401, detail="Invalid token: invalid audience")


async def get_current_user(authorization: str | None = Header(None)) -> dict[str, Any]:
    """
    Extract and validate user from JWT token.

    Production: Clerk issues RS256 JWTs. We verify using PyJWKClient against
    CLERK_JWKS_URL — the JWKS endpoint contains the RSA public keys needed to
    verify the signature. CLERK_SECRET_KEY is a backend API secret, NOT a JWT
    signing key, so it must NOT be used for jwt.decode().

    Local dev (no CLERK_JWKS_URL): signature verification is skipped so developers
    can use a minimal JWT without a Clerk account. Never set ENVIRONMENT=production
    without CLERK_JWKS_URL configured.
    """
    if not authorization:
        raise HTTPException(status_code=401, detail="Missing authorization header")

    if not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Invalid authorization format")

    token = authorization.removeprefix("Bearer ")

    try:
        if _jwks_client:
            # Production path: RS256 verification via Clerk JWKS
            signing_key = _jwks_client.get_signing_key_from_jwt(token)
            jwt_decode_kwargs: dict[str, Any] = {
                "algorithms": ["RS256"],
                "options": {"verify_exp": True},
            }
            if settings.CLERK_EXPECTED_ISSUER:
                jwt_decode_kwargs["issuer"] = settings.CLERK_EXPECTED_ISSUER
            if settings.CLERK_EXPECTED_AUDIENCE:
                jwt_decode_kwargs["audience"] = settings.CLERK_EXPECTED_AUDIENCE
            payload = jwt.decode(
                token,
                signing_key.key,
                **jwt_decode_kwargs,
            )
        else:
            # Local dev path: skip signature verification entirely
            # Requires ENVIRONMENT != "production" to prevent accidental misconfiguration
            if settings.ENVIRONMENT == "production":
                logger.error(
                    "auth.jwks_url_missing_in_production",
                    hint="Set CLERK_JWKS_URL to your Clerk JWKS endpoint",
                )
                raise HTTPException(
                    status_code=500,
                    detail="Server misconfiguration: CLERK_JWKS_URL not set in production",
                )
            logger.debug("auth.dev_mode_no_signature_check")
            payload = jwt.decode(token, options={"verify_signature": False})
            _validate_optional_clerk_claims(payload)

        user_id = payload.get("sub")
        if not user_id:
            raise HTTPException(status_code=401, detail="Invalid token: missing sub claim")

        logger.info("auth.user_authenticated", user_id=str(user_id)[:20])

        return {
            "user_id": user_id,
            "tier": payload.get("googly_tier", "free"),
        }
    except jwt.ExpiredSignatureError:
        logger.warning("auth.token_expired")
        raise HTTPException(status_code=401, detail="Token has expired")
    except jwt.PyJWKClientError as e:
        logger.warning("auth.jwks_fetch_error", error=str(e))
        raise HTTPException(status_code=401, detail="Could not verify token signature")
    except jwt.InvalidTokenError as e:
        logger.warning("auth.token_invalid", error=str(e))
        raise HTTPException(status_code=401, detail=f"Invalid token: {str(e)}")
    except HTTPException:
        raise
    except Exception:
        logger.exception("auth.unexpected_error")
        raise HTTPException(status_code=401, detail="Authentication failed")


async def verify_internal_token(x_internal_token: str | None = Header(None)) -> bool:
    """
    Verify X-Internal-Token to ensure request came from APIM.

    In local dev, this is skipped if no token is configured.
    In production, this must match APIM_INTERNAL_TOKEN.
    """
    if not settings.APIM_INTERNAL_TOKEN:
        logger.debug("auth.internal_token_skipped", reason="not_configured")
        return True

    if not x_internal_token:
        logger.warning("auth.internal_token_missing")
        raise HTTPException(status_code=403, detail="Missing X-Internal-Token header")

    if not hmac.compare_digest(x_internal_token, settings.APIM_INTERNAL_TOKEN):
        logger.warning("auth.internal_token_invalid")
        raise HTTPException(status_code=403, detail="Invalid X-Internal-Token")

    logger.debug("auth.internal_token_verified")
    return True


CurrentUser = Annotated[dict, Depends(get_current_user)]
VerifiedInternal = Annotated[bool, Depends(verify_internal_token)]
