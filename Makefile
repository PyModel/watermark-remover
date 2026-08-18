.PHONY: check lint format-check format lint-fix compile test smoke smoke-synthid bootstrap-synthid docker-synthid-build docker-synthid-help \
	smoke-ctrlregen bootstrap-ctrlregen docker-ctrlregen-build docker-ctrlregen-help \
	smoke-markllm bootstrap-markllm docker-markllm-build docker-markllm-help \
	smoke-markdiffusion bootstrap-markdiffusion docker-markdiffusion-build docker-markdiffusion-help \
	docker-core-build docker-core-help serve compose-up compose-up-heavy compose-check \
	demo install-skill install-cursor-text-skill clean

SCRIPTS := skills/remove-ai-marks/scripts
PYTHON ?= $(shell if [ -x .venv/bin/python ]; then echo .venv/bin/python; else echo python3; fi)

check: lint format-check compile test smoke

lint:
	$(PYTHON) -m ruff check $(SCRIPTS) tests demo.py

# Upstream-compatible alias: `format` is a check, not a formatter.
format-check:
	$(PYTHON) -m ruff format --check $(SCRIPTS) tests demo.py

format: format-check

lint-fix:
	$(PYTHON) -m ruff check --fix $(SCRIPTS) tests demo.py

compile:
	$(PYTHON) -m compileall -q $(SCRIPTS) demo.py

test:
	$(PYTHON) -m pytest

smoke:
	-$(PYTHON) $(SCRIPTS)/inspect_text.py tests/fixtures/sample_watermarked.txt
	$(PYTHON) $(SCRIPTS)/clean_text.py tests/fixtures/sample_watermarked.txt -o /tmp/wm.cleaned.txt --stats
	$(PYTHON) $(SCRIPTS)/rewrite_text.py tests/fixtures/sample_watermarked.txt --backend print-prompt --strength paraphrase >/dev/null
	$(PYTHON) $(SCRIPTS)/rewrite_text.py tests/fixtures/sample_watermarked.txt --backend print-prompt --strength tsapa --generations 2 --population 4 >/dev/null
	$(PYTHON) $(SCRIPTS)/perturb_text.py tests/fixtures/sample_watermarked.txt --mode zero-width --strength 0.1 --seed 1 -o /tmp/wm.perturbed.txt
	-$(PYTHON) $(SCRIPTS)/inspect_file.py tests/fixtures/sample_ai.md
	$(PYTHON) $(SCRIPTS)/clean_file.py tests/fixtures/sample_ai.md -o /tmp/sample_ai.cleaned.md
	$(PYTHON) $(SCRIPTS)/clean_file.py tests/fixtures/sample_ai.html -o /tmp/sample_ai.cleaned.html
	$(PYTHON) $(SCRIPTS)/clean_file.py tests/fixtures/sample_meta.svg -o /tmp/sample_meta.cleaned.svg
	$(PYTHON) $(SCRIPTS)/inspect_soft_binding.py tests/fixtures/sample_meta.svg >/dev/null
	@echo "smoke ok"

smoke-synthid:
	@if [ -z "$(REVERSE_SYNTHID_DIR)" ]; then \
	  echo "smoke-synthid skipped (set REVERSE_SYNTHID_DIR)"; \
	else \
	  $(PYTHON) $(SCRIPTS)/score_synthid.py --help >/dev/null && echo "score_synthid adapter present"; \
	fi

bootstrap-synthid:
	./skills/remove-ai-marks/scripts/setup_synthid.sh

docker-synthid-build:
	docker build -f Dockerfile.synthid -t watermark-remover-synthid-scorer .

docker-synthid-help:
	docker run --rm watermark-remover-synthid-scorer --help

# Optional CtrlRegen pixel-removal backend (external noai-watermark checkout).
smoke-ctrlregen:
	@if [ -z "$(NOAI_WATERMARK_DIR)" ]; then \
	  echo "smoke-ctrlregen skipped (set NOAI_WATERMARK_DIR)"; \
	else \
	  $(PYTHON) $(SCRIPTS)/clean_ctrlregen.py --help >/dev/null && echo "clean_ctrlregen adapter present"; \
	fi

bootstrap-ctrlregen:
	./skills/remove-ai-marks/scripts/setup_ctrlregen.sh

docker-ctrlregen-build:
	docker build -f Dockerfile.ctrlregen -t watermark-remover-ctrlregen .

docker-ctrlregen-help:
	docker run --rm watermark-remover-ctrlregen --help

# Optional MarkLLM text-watermark harness (external checkout).
smoke-markllm:
	@if [ -z "$(MARKLLM_DIR)" ]; then \
	  echo "smoke-markllm skipped (set MARKLLM_DIR)"; \
	else \
	  $(PYTHON) $(SCRIPTS)/detect_text_watermark.py --help >/dev/null && echo "detect_text_watermark adapter present"; \
	fi

bootstrap-markllm:
	./skills/remove-ai-marks/scripts/setup_markllm.sh

docker-markllm-build:
	docker build -f Dockerfile.markllm -t watermark-remover-markllm .

docker-markllm-help:
	docker run --rm watermark-remover-markllm --help

# Optional MarkDiffusion image-watermark harness (PyPI or external checkout).
smoke-markdiffusion:
	@if [ -z "$(MARKDIFFUSION_DIR)" ]; then \
	  echo "smoke-markdiffusion skipped (set MARKDIFFUSION_DIR)"; \
	else \
	  $(PYTHON) $(SCRIPTS)/markdiffusion_harness.py --help >/dev/null && echo "markdiffusion_harness adapter present"; \
	fi

bootstrap-markdiffusion:
	./skills/remove-ai-marks/scripts/setup_markdiffusion.sh

docker-markdiffusion-build:
	docker build -f Dockerfile.markdiffusion -t watermark-remover-markdiffusion .

docker-markdiffusion-help:
	docker run --rm watermark-remover-markdiffusion --help

# Core HTTP service (text + file/image metadata cleaning).
docker-core-build:
	docker build -f Dockerfile -t watermark-remover .

docker-core-help:
	docker run --rm watermark-remover wm-serve --help

# Run the HTTP service locally (stdlib only, no Docker).
serve:
	$(PYTHON) $(SCRIPTS)/server.py --host 127.0.0.1 --port 8765

compose-up:
	docker compose up --build -d

compose-up-heavy:
	docker compose --profile harness --profile heavy up --build -d

compose-check:
	./compose-check.sh

demo:
	$(PYTHON) demo.py

install-skill:
	mkdir -p $(HOME)/.grok/skills
	ln -sfn $(CURDIR)/skills/remove-ai-marks $(HOME)/.grok/skills/remove-ai-marks
	@echo "linked -> $(HOME)/.grok/skills/remove-ai-marks"

install-cursor-text-skill:
	$(PYTHON) install_skill.py

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	rm -rf .pytest_cache .venv
