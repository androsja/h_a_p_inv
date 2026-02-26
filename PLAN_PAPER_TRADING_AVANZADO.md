# 🏆 Plan de Desarrollo: "Hapi Shadow Broker" (Simulador Institucional)

Tu objetivo es impecable: Necesitamos un simulador ("Paper Trading") que sea un **"Clon Espejo" (Shadow)** perfecto de las condiciones comerciales, tarifas y comisiones de HapiTrade. Si el bot es rentable en este simulador hostil de "mundo real", encenderlo con dinero de verdad será un mero trámite administrativo. 

Este plan transforma nuestro actual motor ciego (que asume que hacer trading es gratis) en un **ecosistema financiero rudo y realista**, completo con "Slippage", retenciones en tránsito (T+1) y cobros de mantenimiento/retiro.

---

## 🎯 Objetivo General
Desarrollar un entorno de *Paper Trading en Vivo (Shadow Broker)* que reproduzca fielmente las comisiones ("Fees"), spreads (diferencial bid-ask) y tiempos de compensación de HapiTrade. El usuario debe poder "Fondear", "Retirar" y observar cómo la plataforma (Hapi) se alimenta de las comisiones en cada operación, y aun así, comprobar si el Algoritmo de IA es rentable.

---

## 🧱 FASE 1: Motor Contable Realista (El "Cajero" de Hapi)
*Antes de conectar datos vivos, nuestra clase `AccountState` necesita aprender a cobrar exactamente los mismos impuestos que el broker real.*

### 1.1 Modelo de Comisiones Oficiales de Hapi
He investigado el listado exacto de cobros actuales de HapiTrade. El algoritmo descontará **matemáticamente lo mismo que la vida real**:
*   **Comisión de Cierre (Trade de Venta):** Hapi cobra **$0.10** por venta de acciones enteras y **$0.15** si es una fracción de acción. (Para Crypto cobran 1% fijo).
*   **Tarifa de la SEC (Solo Ventas):** El Gobierno de EEUU te cobra **$0.0000278 por cada $1.00 USD** de capital vendido (Min $0.01).
*   **Tarifa FINRA (TAF):** Se cobra **$0.000166 por cada acción** vendida (Redondeado al centavo más cercano).
*   **Registro Contable Real:** El sistema actual (`state.json`) ahora tendrá: 
    *   `total_fees_paid`: Historial de todo el dinero entregado a Hapi y a la SEC.
    *   `net_profit`: PnL Real (Ganancia bruta - (Comisión de Cierre + SEC + FINRA + Deslizamiento)).

### 1.2 "Slippage" (Deslizamiento de Precios)
Hapi no cobra Spread por acciones y ETFs, pero en la vida real, si la IA presiona comprar a $100.00 por red wifi desde Latam, Hapi podría ejecutarte a $100.02.
*   Modificaremos el Simulador para que **añada un retraso virtual de 300 ms** y deslice (empeore) el precio de compra en un `0.05%` aleatorio, replicando la latencia de red contra Wall Street. ¡Si el bot sobrevive al *Slippage* y a los cobros regulatorios, sobrevivirá al mercado real!

---

## 🏦 FASE 2: Gestión de Tesorería (Cajero Automático)
*El bot necesita un banco. Tú debes poder sacar o meter plata al ecosistema en caliente.*

### 2.1 API de Tesorería
Crearemos "Endpoints" en nuestro backend (`server.py`) que imitarán a tu banco:
*   `POST /api/bank/deposit`: Para inyectarle, por ejemplo, $500.00 al saldo del algoritmo.
*   `POST /api/bank/withdraw`: Para pedirle un retiro de $100.00 al bot.

### 2.2 Custodia y Settlement (Liquidación T+1)
Para hacerlo ridículamente realista:
*   En Hapi, si vendes una acción, el dinero no se puede retirar inmediatamente al banco; tarda 1 día (Settlement T+1).
*   Programaremos una "Billetera Virtual" con dos estados: `Cash Disponible para Invertir` y `Cash Liquidado (Retirable al banco)`.

### 2.3 Botón Financiero en el Frontend (Web)
*   Agregaremos un modal elegante: **"🏦 Hapi Bank Simulator"** en la interfaz. Podrás inyectar saldo. Si el bot gana, presionarás "Retirar" y **Hapi te cobrará la comisión plana de retiro que es de $4.99 USD** por transferencias hacia Latam. Mostrará un recibo de lo depositado "en el banco".

---

## 📡 FASE 3: El Ojo de Sauron (Datos en Vivo)
*El simulador realista debe poder operar con la vela del minuto actual, no del mes pasado.*

### 3.1 Motor de "Ticks" (Data Feed en Vivo)
*   En el archivo `main.py`, al modo `LIVE_PAPER`, pasaremos del loop infinito de *backtesting* a un loop controlado por reloj cronológico (`time.sleep(60)`).
*   Cada 60 segundos ("Tick"), consultaremos la **última vela real de Yahoo Finance (`yfinance`) o Polygon (Gratis)**.
*   La IA (`ai_model.joblib`) ingerirá la vela a las 09:31:00 AM (Nueva York), pensará por milisegundos, y tomará la decisión en tiempo real. 

---

## 💻 FASE 4: Comando y Control UI (El Dropdown)
*Unir todo mecánicamente desde la Web.*

### 4.1 Selector de Combate (`index.html` → `command.json`)
*   Le daremos vida al **Dropdown "Seleccionar Símbolo"**. 
*   Haremos que lea de tu `assets.json` las empresas activas, ej: `(AAPL, MSFT, NVDA)`.
*   Botón de Play rojo: **"🚀 Lanzar Escuadrón (Live Paper)"**. 

### 4.2 Arquitectura Final de Ejecución:
1. Tú seleccionas "NVDA" y presionas Iniciar (Live Paper).
2. El Dashboard detiene el simulador ciego (`docker restart`).
3. El motor `main.py` despierta, lee tus credenciales ficticias y arranca.
4. Consulta el saldo, compra en vivo y **paga las comisiones virtuales de Hapi**.
5. Te envía notificaciones visuales en el log ("💎 Ganancia de $10.00 pero Hapi descontó $0.15").

---

## 📅 Cronograma Propuesto de Implementación
*   **Paso 1 (Mañana):** Lógica Contable. Re-escribir el Simulador para que descuente comisiones por trade, aplique Slippage, y lo muestre en consola.
*   **Paso 2:** Módulo Financiero. Agregar endpoints de Depósito/Retiro y programar el estado del banco simulado (T+1).
*   **Paso 3:** Conexión de Precios Vivos y Front-End. Crear el Dropdown de "Activos" y las gráficas en vivo que consumen velas nuevas cada minuto.

*¡Esta es la verdadera forma de probar un Algoritmo sin sangrar dinero en bolsa! ¿Qué opinas del esquema contable de comisiones del Paso 1? ¿Estás listo para iniciar la programación?*
