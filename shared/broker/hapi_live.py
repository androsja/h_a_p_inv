"""
broker/hapi_live.py ─ Implementación Live de la API de Hapi Trade.

Características:
  • Autenticación con API_KEY + CLIENT_ID + USER_TOKEN (Bearer)
  • Soporte de modo sandbox vía cabecera IsTestMode
  • Manejo automático de errores 429 (Rate Limit) con back-off exponencial
  • Reintento automático de órdenes rechazadas hasta MAX_RETRIES veces
  • Gestión de sesión HTTP con headers reutilizables
"""

import time
import uuid
import requests
from datetime import datetime, timezone

import config
from broker.interface import BrokerInterface, Quote, OrderResponse, AccountInfo
from utils.logger import log, log_rate_limit


# ─── Configuración de reintentos ─────────────────────────────────────────────
MAX_RETRIES    = 3
RETRY_WAIT_SEC = 2      # Espera base entre reintentos (se duplica en cada intento)
TIMEOUT_SEC    = 5      # Tiempo máximo de espera por respuesta de la API


class HapiLive(BrokerInterface):
    """
    Conector en vivo con la API REST de Hapi Trade.

    Hapi Trade es un bróker para el mercado latinoamericano que permite
    depósitos por PSE y opera a través de Apex Clearing.

    Endpoints principales (documentación interna de Hapi):
      POST /auth/token         → Refrescar token de sesión
      GET  /quotes/{symbol}    → Cotización en tiempo real
      POST /orders             → Crear nueva orden
      GET  /account            → Estado de la cuenta
      DELETE /orders/{id}      → Cancelar orden
    """

    def __init__(self, api_key: str, client_id: str, user_token: str):
        if not all([api_key, client_id, user_token]):
            raise ValueError(
                "HapiLive requiere API_KEY, CLIENT_ID y USER_TOKEN. "
                "¿Olvidaste configurar el archivo .env?"
            )
        self._api_key    = api_key
        self._client_id  = client_id
        self._user_token = user_token
        self._base_url   = config.HAPI_BASE_URL
        self._session    = self._build_session()
        log.info(
            f"HapiLive | Conectado{'  [SANDBOX/TEST]' if config.HAPI_TEST_MODE else '  ⚠️  [DINERO REAL]'}"
        )

    # ─── Propiedades ─────────────────────────────────────────────────────────
    @property
    def name(self) -> str:
        return "HapiLive"

    @property
    def is_paper_trading(self) -> bool:
        return config.HAPI_TEST_MODE

    # ─── Sesión HTTP ─────────────────────────────────────────────────────────
    def _build_session(self) -> requests.Session:
        """Crea una sesión HTTP con los headers de autenticación de Hapi."""
        session = requests.Session()
        session.headers.update({
            "Content-Type":  "application/json",
            "Accept":        "application/json",
            "X-Api-Key":     self._api_key,
            "X-Client-Id":   self._client_id,
            "Authorization": f"Bearer {self._user_token}",
            # ── Modo sandbox de Hapi: las órdenes se procesan pero NO
            #    usan dinero real cuando este header es "true".
            "IsTestMode":    "true" if config.HAPI_TEST_MODE else "false",
        })
        return session

    # ─── Llamada base a la API con reintentos y manejo de 429 ────────────────
    def _request(
        self,
        method: str,
        endpoint: str,
        **kwargs,
    ) -> dict:
        """
        Realiza una solicitud HTTP al API de Hapi con:
          • Timeout de TIMEOUT_SEC segundos
          • Reintentos con back-off exponencial
          • Manejo especial del error 429 (Rate Limit)
        """
        url = f"{self._base_url}{endpoint}"
        wait = RETRY_WAIT_SEC

        for attempt in range(1, MAX_RETRIES + 1):
            try:
                resp = self._session.request(
                    method, url, timeout=TIMEOUT_SEC, **kwargs
                )

                # ── Rate Limit ──────────────────────────────────────────────
                if resp.status_code == 429:
                    retry_after = resp.headers.get("Retry-After")
                    retry_secs  = int(retry_after) if retry_after else wait
                    log_rate_limit(endpoint, retry_secs)
                    time.sleep(retry_secs)
                    wait *= 2       # Back-off exponencial
                    continue

                # ── Errores de servidor ──────────────────────────────────────
                if resp.status_code >= 500:
                    log.error(
                        f"HapiLive | {method} {endpoint} → HTTP {resp.status_code} "
                        f"(intento {attempt}/{MAX_RETRIES})"
                    )
                    time.sleep(wait)
                    wait *= 2
                    continue

                resp.raise_for_status()
                return resp.json()

            except requests.exceptions.Timeout:
                log.warning(
                    f"HapiLive | Timeout en {endpoint} (intento {attempt}/{MAX_RETRIES})"
                )
                time.sleep(wait)
                wait *= 2

            except requests.exceptions.ConnectionError as exc:
                log.error(f"HapiLive | Error de conexión: {exc}")
                raise ConnectionError(
                    "No se pudo conectar con Hapi Trade. Verifica tu conexión a internet."
                ) from exc

        raise TimeoutError(
            f"HapiLive | {endpoint} falló después de {MAX_RETRIES} intentos."
        )

    # ─── Implementación de BrokerInterface ───────────────────────────────────
    def get_quote(self, symbol: str) -> Quote:
        """Obtiene la cotización en tiempo real desde Hapi."""
        log.debug(f"HapiLive | Solicitando quote para {symbol}")
        data = self._request("GET", f"/quotes/{symbol}")

        return Quote(
            symbol=symbol,
            bid=float(data["bid"]),
            ask=float(data["ask"]),
            last=float(data.get("last", data["ask"])),
            timestamp=data.get("timestamp", datetime.now(timezone.utc).isoformat()),
        )

    def place_limit_order(
        self,
        symbol: str,
        side: str,
        limit_price: float,
        qty: float,
    ) -> OrderResponse:
        """
        Envía una orden límite a Hapi Trade.

        El parámetro 'isTestOrder' en el body es un segundo nivel de protección
        adicional al header IsTestMode, recomendado por la documentación de Hapi.
        """
        payload = {
            "clientOrderId": str(uuid.uuid4()),
            "symbol":        symbol,
            "side":          side,            # "BUY" | "SELL"
            "orderType":     "LIMIT",
            "limitPrice":    limit_price,
            "quantity":      qty,
            "timeInForce":   "DAY",           # Orden válida solo el día actual
            "isTestOrder":   config.HAPI_TEST_MODE,
        }

        log.debug(f"HapiLive | Enviando orden: {payload}")
        try:
            data = self._request("POST", "/orders", json=payload)
            return OrderResponse(
                order_id=data.get("orderId", "UNKNOWN"),
                symbol=symbol,
                side=side,
                status=data.get("status", "PENDING"),
                limit_price=limit_price,
                qty=qty,
                fill_price=data.get("fillPrice"),
                message=data.get("message", ""),
            )
        except Exception as exc:
            log.error(f"HapiLive | Error colocando orden {side} {symbol}: {exc}")
            return OrderResponse(
                order_id="ERROR",
                symbol=symbol,
                side=side,
                status="REJECTED",
                limit_price=limit_price,
                qty=qty,
                fill_price=None,
                message=str(exc),
            )

    def get_account_info(self) -> AccountInfo:
        """Obtiene el estado actual de la cuenta desde Hapi."""
        data = self._request("GET", "/account")
        return AccountInfo(
            total_cash=float(data.get("totalCash", 0)),
            buying_power=float(data.get("buyingPower", 0)),
            portfolio_value=float(data.get("portfolioValue", 0)),
            day_traded_count=int(data.get("dayTradedCount", 0)),
        )

    def cancel_all_orders(self, symbol: str) -> bool:
        """Cancela todas las órdenes abiertas para el símbolo dado."""
        try:
            self._request("DELETE", f"/orders?symbol={symbol}&status=OPEN")
            log.info(f"HapiLive | Órdenes abiertas de {symbol} canceladas.")
            return True
        except Exception as exc:
            log.error(f"HapiLive | No se pudo cancelar las órdenes de {symbol}: {exc}")
            return False
