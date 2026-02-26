# ══════════════════════════════════════════════════════════════
#  Makefile — Comandos rápidos para el Hapi Scalping Bot
#  Uso: make <comando>
# ══════════════════════════════════════════════════════════════

.PHONY: help setup build sim live logs stop clean test

# ── Colores ─────────────────────────────────────────────────────────────────
GREEN  = \033[0;32m
YELLOW = \033[0;33m
CYAN   = \033[0;36m
RESET  = \033[0m

help: ## Muestra esta ayuda
	@echo ""
	@echo "$(CYAN)╔══════════════════════════════════════════════╗$(RESET)"
	@echo "$(CYAN)║    🇨🇴 Hapi Scalping Bot  •  Comandos        ║$(RESET)"
	@echo "$(CYAN)╚══════════════════════════════════════════════╝$(RESET)"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  $(GREEN)%-12s$(RESET) %s\n", $$1, $$2}'
	@echo ""

setup: ## Copia .env.example a .env (primera vez)
	@if [ ! -f .env ]; then \
		cp .env.example .env; \
		echo "$(YELLOW)✅ .env creado. Rellena tus credenciales de Hapi antes de correr en LIVE.$(RESET)"; \
	else \
		echo "$(YELLOW)ℹ️  .env ya existe. No se sobreescribió.$(RESET)"; \
	fi

build: ## Construye la imagen Docker
	@echo "$(CYAN)🔨 Construyendo imagen Docker...$(RESET)"
	docker compose build

sim: build ## Inicia el bot en modo SIMULADO (sin dinero real)
	@echo "$(GREEN)🔵 Iniciando modo SIMULADO...$(RESET)"
	docker compose up trading-bot-sim

sim-detached: build ## Inicia el modo SIMULADO en segundo plano
	@echo "$(GREEN)🔵 Iniciando modo SIMULADO en background...$(RESET)"
	docker compose up -d trading-bot-sim

live: build ## ⚠️  Inicia el bot en modo LIVE (dinero real — requiere .env configurado)
	@echo "$(YELLOW)⚠️  ADVERTENCIA: Modo LIVE usa DINERO REAL.$(RESET)"
	@read -p "¿Estás seguro? (escribe 'SI' para continuar): " confirm; \
		[ "$$confirm" = "SI" ] && docker compose --profile live up trading-bot-live || echo "Cancelado."

logs: ## Ver los logs en tiempo real
	docker compose logs -f

logs-file: ## Ver el archivo de log del bot
	@tail -f logs/trading_bot.log 2>/dev/null || echo "ℹ️  No hay logs todavía. Corre 'make sim' primero."

stop: ## Detiene todos los contenedores del bot
	@echo "$(YELLOW)🛑 Deteniendo bot...$(RESET)"
	docker compose down

clean: stop ## Detiene y elimina contenedores, imágenes y volúmenes de caché
	@echo "$(YELLOW)🧹 Limpiando...$(RESET)"
	docker compose down --rmi local --volumes
	@rm -rf data/cache/ logs/
	@echo "$(GREEN)✅ Limpieza completa.$(RESET)"

test: ## Verifica que los módulos del bot importan correctamente (sin Docker)
	@echo "$(CYAN)🧪 Verificando imports...$(RESET)"
	@python3 -c " \
		import sys; sys.path.insert(0, '.'); \
		from config import TRADING_MODE; \
		from utils.market_hours import market_status_str; \
		print('  ✅ config      OK'); \
		print('  ✅ utils       OK'); \
		print('  ✅ Estado:', market_status_str()); \
	" 2>&1 || echo "$(YELLOW)⚠️  Falta instalar dependencias. Corre 'make build' para usar Docker.$(RESET)"

status: ## Muestra el estado de los contenedores del bot
	docker compose ps
