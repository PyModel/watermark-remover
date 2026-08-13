.PHONY: check lint format-check compile test smoke smoke-synthid bootstrap-synthid docker-synthid-build docker-synthid-help demo install-skill clean

SCRIPTS := skills/remove-ai-marks/scripts
PYTHON ?= $(shell if [ -x .venv/bin/python ]; then echo .venv/bin/python; else echo python3; fi)

check: lint format-check compile test smoke

lint:
	$(PYTHON) -m ruff check $(SCRIPTS) tests demo.py

format-check:
	$(PYTHON) -m ruff format --check $(SCRIPTS) tests demo.py

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

demo:
	$(PYTHON) demo.py

install-skill:
	mkdir -p $(HOME)/.grok/skills
	ln -sfn $(CURDIR)/skills/remove-ai-marks $(HOME)/.grok/skills/remove-ai-marks
	@echo "linked -> $(HOME)/.grok/skills/remove-ai-marks"

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	rm -rf .pytest_cache .venv
