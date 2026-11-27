# TAP training on the server.
#
#   make train      update the code, then start the TAP training in the background (scripts/train_tap_f1.sh):
#                   1. fold roadmap of recording FILES, gesture aware (fold_split.py; people are never a split unit)
#                   2. signal labels + inputs for every DATA_TAP recording (build_tap_f1_dataset.py -> DATA/DATA_TAP_F1)
#                   3. the legacy F1 trainer, unchanged but for the data / roadmap paths and one line that drops
#                      unknown rows (train_tap_f1.py: Optuna LightGBM + XGBoost, F1 at 0.85 + size, max 96 KB)
#                   4. event check of every saved model (evaluate_tap_f1.py -> TAP_MODELS_F1_FILEFOLDS/event_eval.md)
#                   A previous TAP_MODELS_F1_FILEFOLDS is renamed .done_<time>; nothing is overwritten.
#   make status     is it running, which step, how many trials, last lines
#   make stop       stop every running training (also ones started by hand)
#   make smoke      the whole chain on 2 recordings per folder and fold, 3 trials, in /tmp (a few minutes)
#
# Settings (defaults measured in research/gesture/TAP_F1_LEGACY.md):
#   LABELS=legacy      signal labels: the top of every tap is 1, every other row 0; pointer repaired (late / early
#                      pointer, bounces, the 2nd tap of a double, taps the pointer missed); broken recordings left out
#   LABELS=v10rescue   your old labeller, unchanged, on the repaired pointer
#   LABELS=v10         your old labeller exactly
#   FEATURES=newacc    the raw fields + the legacy model's 64 diff / ratio inputs + 31 ACC ratios (default)
#   FEATURES=leg       without the ACC ratios;  FEATURES=new  + EMG diffs;  FEATURES=lite  + Lean-lite (needs WL=1)
#   WL=0               no EMG waveform-length (WL) input at all (default; on the device WL-led models did worse);
#                      WL=1 allows the 9 WL fields and WL ratios again
#   TAPROWS=0.5        TAP / SWIPE recordings train on every 2nd row: the legacy model's data mix, half the rows from
#                      movement recordings (today's data has a third; all rows: TAPROWS=1)
#   TRIALS=100         Optuna trials per model family
#   make train LABELS=v10 FEATURES=leg TAPROWS=1    exactly the old pipeline's labels, inputs and mix, on file folds
#
#   make train-unified RECIPE=final14top KB= ALT_KB= TARGET= FIST= CERTAIN= LIMIT= EXTRA=
#                   the previous pipeline (research/gesture/train_unified.py; recipes and settings in
#                   research/gesture/PIPELINE.md); make status / stop cover both
RECIPE ?= final14top
LIMIT  ?=
override SPLIT := file# fixed: splits and folds by recording file only (a person split is not allowed)
ALT_KB ?= 10
EXTRA  ?=
PYTHON ?= python
PULL   ?= 1
TARGET ?=
KB     ?=
CERTAIN ?=
FIST   ?=
# tree-count range for KB: up to 25 trees per KB (at least 400), so small trees can fill a big budget
COMMA  := ,
NEST    = 80$(COMMA)$(shell echo $$(( $(KB) * 25 > 400 ? $(KB) * 25 : 400 )))
FORCE  ?= 0

SCRIPT := research/gesture/train_unified.py
LOGDIR := logs
LOG    := $(LOGDIR)/tap_$(RECIPE)_split$(SPLIT)_$(shell date +%Y%m%d_%H%M%S).log
PIDFILE := $(LOGDIR)/train.pid
# a python process running the trainer (anchored, so shells and editors that mention the file do not match)
TRAIN_RE := ^[^ ]*python[0-9.]* .*(train_unified|train_tap_f1|build_tap_f1_dataset|evaluate_tap_f1)\.py|^(ba)?sh [^ ]*train_tap_f1\.sh
LABELS   ?= legacy
FEATURES ?= newacc
TAPROWS  ?= 0.5
WL       ?= 0
TRIALS   ?= 100
F1_LOG   := $(LOGDIR)/tap_f1_$(LABELS)_$(FEATURES)_rows$(TAPROWS)_wl$(WL)_$(shell date +%Y%m%d_%H%M%S).log

.PHONY: train train-unified status stop smoke

train:
	@running=$$(pgrep -af '$(TRAIN_RE)'); \
	if [ -n "$$running" ] && [ "$(FORCE)" != "1" ]; then \
	  echo "A training is already running:"; echo "$$running"; \
	  echo "Wait for it to finish (make status), or start anyway with: make train FORCE=1"; exit 1; fi
	@if [ "$(PULL)" = "1" ]; then git pull --ff-only; fi
	@mkdir -p $(LOGDIR)
	@TAP_F1_LABELS=$(LABELS) TAP_F1_FEATURES=$(FEATURES) TAP_F1_TAP_ROW_SHARE=$(TAPROWS) TAP_F1_WL=$(WL) LEGACY_N_TRIALS=$(TRIALS) PYTHON=$(PYTHON) \
	  nohup bash scripts/train_tap_f1.sh > $(F1_LOG) 2>&1 & echo $$! > $(PIDFILE)
	@ln -sfn $(notdir $(F1_LOG)) $(LOGDIR)/latest.log
	@echo "Started: TAP F1 pipeline labels=$(LABELS) features=$(FEATURES) tap_rows=$(TAPROWS) wl=$(WL) trials=$(TRIALS)"
	@echo "PID $$(cat $(PIDFILE)), log $(F1_LOG)"
	@sleep 20; \
	if kill -0 $$(cat $(PIDFILE)) 2>/dev/null; then \
	  echo "Running. Check progress with: make status"; \
	else \
	  echo "The training stopped right away. Last lines of the log:"; tail -n 30 $(F1_LOG); exit 1; fi

smoke:
	@rm -rf /tmp/tap_f1_smoke && mkdir -p /tmp/tap_f1_smoke/DATA
	$(PYTHON) fold_split.py build
	cp DATA/fold_roadmap_files.json /tmp/tap_f1_smoke/DATA/
	$(PYTHON) build_tap_f1_dataset.py --out /tmp/tap_f1_smoke/DATA/DATA_TAP_F1 --limit 2 --labels $(LABELS) --features $(FEATURES) --tap-row-share $(TAPROWS) --wl $(WL)
	cd /tmp/tap_f1_smoke && LEGACY_N_TRIALS=3 $(PYTHON) -u $(CURDIR)/train_tap_f1.py
	$(PYTHON) evaluate_tap_f1.py /tmp/tap_f1_smoke/TAP_MODELS_F1_FILEFOLDS --data /tmp/tap_f1_smoke/DATA/DATA_TAP_F1
	@echo "smoke OK: /tmp/tap_f1_smoke/TAP_MODELS_F1_FILEFOLDS"

train-unified:
	@if [ "$(RECIPE)" = "final" ]; then echo "RECIPE=final is a FAILED experiment (no mid / strong taps on the band, 2026-11-26). Use the default."; exit 1; fi
	@for kb in $(KB) $(ALT_KB); do if awk -v k="$$kb" 'BEGIN{exit !(k * 1024 > 102400)}'; then \
	  echo "$$kb KB exceeds the device maximum MAX_MODEL_SIZE_BYTES = 102400 (100 KB)"; exit 1; fi; done
	@running=$$(pgrep -af '$(TRAIN_RE)'); \
	if [ -n "$$running" ] && [ "$(FORCE)" != "1" ]; then \
	  echo "A training is already running:"; echo "$$running"; \
	  echo "Wait for it to finish (make status), or start anyway with: make train FORCE=1"; exit 1; fi
	@if [ "$(PULL)" = "1" ]; then git pull --ff-only; fi
	@MUDRA_TAP_RECIPE=$(RECIPE) MUDRA_SPLIT=$(SPLIT) MUDRA_ALT_BUDGET_KB=$(ALT_KB) $(if $(TARGET),MUDRA_TAP_FALSE_PER_MIN_TARGET=$(TARGET)) $(if $(LIMIT),MUDRA_TAP_MOVEMENT_LIMIT=$(LIMIT)) $(if $(CERTAIN),MUDRA_TAP_CERTAIN=$(CERTAIN)) $(if $(FIST),MUDRA_TAP_FIST_WEIGHT=$(FIST)) $(if $(KB),MUDRA_MODEL_BUDGET_KB=$(KB) MUDRA_N_ESTIMATORS=$(NEST)) $(EXTRA) \
	  $(PYTHON) research/gesture/prepare_run.py
	@mkdir -p $(LOGDIR)
	@MUDRA_TAP_RECIPE=$(RECIPE) MUDRA_SPLIT=$(SPLIT) MUDRA_ALT_BUDGET_KB=$(ALT_KB) $(if $(TARGET),MUDRA_TAP_FALSE_PER_MIN_TARGET=$(TARGET)) $(if $(LIMIT),MUDRA_TAP_MOVEMENT_LIMIT=$(LIMIT)) $(if $(CERTAIN),MUDRA_TAP_CERTAIN=$(CERTAIN)) $(if $(FIST),MUDRA_TAP_FIST_WEIGHT=$(FIST)) $(if $(KB),MUDRA_MODEL_BUDGET_KB=$(KB) MUDRA_N_ESTIMATORS=$(NEST)) $(EXTRA) \
	  nohup $(PYTHON) -u $(SCRIPT) > $(LOG) 2>&1 & echo $$! > $(PIDFILE)
	@ln -sfn $(notdir $(LOG)) $(LOGDIR)/latest.log
	@echo "Started: recipe=$(RECIPE) split=$(SPLIT) alt_kb=$(ALT_KB) $(if $(TARGET),target=$(TARGET)) $(if $(LIMIT),limit=$(LIMIT)) $(if $(CERTAIN),certain=$(CERTAIN)) $(if $(FIST),fist_weight=$(FIST)) $(if $(KB),kb=$(KB)) $(EXTRA)"
	@echo "PID $$(cat $(PIDFILE)), log $(LOG)"
	@sleep 20; \
	if kill -0 $$(cat $(PIDFILE)) 2>/dev/null; then \
	  echo "Running. Check progress with: make status"; \
	else \
	  echo "The training stopped right away. Last lines of the log:"; tail -n 30 $(LOG); exit 1; fi

status:
	@running=$$(pgrep -af '$(TRAIN_RE)'); \
	if [ -n "$$running" ]; then echo "Running:"; echo "$$running"; else echo "No training running."; fi
	@if [ -e $(LOGDIR)/latest.log ] && grep -q 'TAP_F1 1/4' $(LOGDIR)/$$(readlink $(LOGDIR)/latest.log); then \
	  log=$(LOGDIR)/$$(readlink $(LOGDIR)/latest.log); \
	  echo "Log: $$log"; \
	  echo "Step: $$(grep 'TAP_F1 ' $$log | tail -n 1)"; \
	  echo "Optuna trials finished: $$(grep -c 'Trial [0-9]* finished' $$log)"; \
	  grep 'Trial [0-9]* finished' $$log | tail -n 2 | cut -c1-200; \
	  if grep -q 'TAP_F1_DONE' $$log; then echo "FINISHED:"; cat TAP_MODELS_F1_FILEFOLDS/event_eval.md 2>/dev/null; fi; \
	  if grep -q 'TAP_F1_FAILED\|Traceback' $$log; then echo "ERROR in the log:"; tail -n 15 $$log; fi; \
	  echo "Last line: $$(tail -n 1 $$log | cut -c1-200)"; \
	elif [ -e $(LOGDIR)/latest.log ]; then \
	  log=$(LOGDIR)/$$(readlink $(LOGDIR)/latest.log); \
	  echo "Log: $$log"; \
	  echo "Trials done: $$(grep -c 'UNIFIED_TRIAL ' $$log)"; \
	  grep 'UNIFIED_TRIAL ' $$log | tail -n 3; \
	  if grep -q 'UNIFIED_BOTH_READY' $$log; then echo "FINISHED:"; grep 'UNIFIED_BOTH_READY' $$log | tail -n 1; fi; \
	  if grep -q 'Traceback' $$log; then echo "ERROR in the log:"; tail -n 15 $$log; fi; \
	  echo "Last line: $$(tail -n 1 $$log | cut -c1-200)"; \
	else echo "No log yet: start with make train"; fi

stop:
	@pids=$$(pgrep -f '$(TRAIN_RE)'); \
	if [ -z "$$pids" ]; then echo "No training running."; exit 0; fi; \
	echo "Stopping:"; pgrep -af '$(TRAIN_RE)'; kill $$pids; \
	for i in 1 2 3 4 5 6 7 8 9 10; do pgrep -f '$(TRAIN_RE)' >/dev/null || break; sleep 1; done; \
	if pgrep -f '$(TRAIN_RE)' >/dev/null; then echo "Still running after 10 s:"; pgrep -af '$(TRAIN_RE)'; exit 1; \
	else echo "Stopped."; fi; rm -f $(PIDFILE)
