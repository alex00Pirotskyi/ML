# TAP training on the server.
#
#   make train      update the code, then start the training in the background
#                   (new recordings: old labels are moved aside and rebuilt; a finished run of the same
#                   settings is renamed <folder>.done_<time>, so every make train trains from scratch)
#   make status     is it running, how many trials, last trial lines
#   make stop       stop every running training (also ones started by hand)
#
# Default: RECIPE=final14obs, the recipe of the model that worked on the band (2026-11-24): file split, 14 KB,
# the old header's false-tap level as the target, no threshold shift. Stricter models (TARGET below 4, RECIPE=final)
# kept their protection offline but lost mid and strong taps on the band (2026-11-25 / 26).
# Settings, for example:
#   RECIPE=final is a FAILED experiment (no mid / strong taps on the band, 2026-11-26): the trainer refuses it
#   make train KB=32 ALT_KB=14          (bigger model, same false-tap level)
#   Model size: at most 100 KB (MAX_MODEL_SIZE_BYTES = 102400, fixed in the trainer)

RECIPE ?= final14obs
LIMIT  ?=
override SPLIT := file# fixed: splits and folds by recording file only (a person split is not allowed)
ALT_KB ?= 10
EXTRA  ?=
PYTHON ?= python
PULL   ?= 1
TARGET ?=
KB     ?=
# tree-count range for KB: up to 25 trees per KB (at least 400), so small trees can fill a big budget
COMMA  := ,
NEST    = 80$(COMMA)$(shell echo $$(( $(KB) * 25 > 400 ? $(KB) * 25 : 400 )))
FORCE  ?= 0

SCRIPT := research/gesture/train_unified.py
LOGDIR := logs
LOG    := $(LOGDIR)/tap_$(RECIPE)_split$(SPLIT)_$(shell date +%Y%m%d_%H%M%S).log
PIDFILE := $(LOGDIR)/train.pid
# a python process running the trainer (anchored, so shells and editors that mention the file do not match)
TRAIN_RE := ^[^ ]*python[0-9.]* .*train_unified\.py

.PHONY: train status stop

train:
	@if [ "$(RECIPE)" = "final" ]; then echo "RECIPE=final is a FAILED experiment (no mid / strong taps on the band, 2026-11-26). Use the default."; exit 1; fi
	@for kb in $(KB) $(ALT_KB); do if awk -v k="$$kb" 'BEGIN{exit !(k * 1024 > 102400)}'; then \
	  echo "$$kb KB exceeds the device maximum MAX_MODEL_SIZE_BYTES = 102400 (100 KB)"; exit 1; fi; done
	@running=$$(pgrep -af '$(TRAIN_RE)'); \
	if [ -n "$$running" ] && [ "$(FORCE)" != "1" ]; then \
	  echo "A training is already running:"; echo "$$running"; \
	  echo "Wait for it to finish (make status), or start anyway with: make train FORCE=1"; exit 1; fi
	@if [ "$(PULL)" = "1" ]; then git pull --ff-only; fi
	@MUDRA_TAP_RECIPE=$(RECIPE) MUDRA_SPLIT=$(SPLIT) MUDRA_ALT_BUDGET_KB=$(ALT_KB) $(if $(TARGET),MUDRA_TAP_FALSE_PER_MIN_TARGET=$(TARGET)) $(if $(LIMIT),MUDRA_TAP_MOVEMENT_LIMIT=$(LIMIT)) $(if $(KB),MUDRA_MODEL_BUDGET_KB=$(KB) MUDRA_N_ESTIMATORS=$(NEST)) $(EXTRA) \
	  $(PYTHON) research/gesture/prepare_run.py
	@mkdir -p $(LOGDIR)
	@MUDRA_TAP_RECIPE=$(RECIPE) MUDRA_SPLIT=$(SPLIT) MUDRA_ALT_BUDGET_KB=$(ALT_KB) $(if $(TARGET),MUDRA_TAP_FALSE_PER_MIN_TARGET=$(TARGET)) $(if $(LIMIT),MUDRA_TAP_MOVEMENT_LIMIT=$(LIMIT)) $(if $(KB),MUDRA_MODEL_BUDGET_KB=$(KB) MUDRA_N_ESTIMATORS=$(NEST)) $(EXTRA) \
	  nohup $(PYTHON) -u $(SCRIPT) > $(LOG) 2>&1 & echo $$! > $(PIDFILE)
	@ln -sfn $(notdir $(LOG)) $(LOGDIR)/latest.log
	@echo "Started: recipe=$(RECIPE) split=$(SPLIT) alt_kb=$(ALT_KB) $(if $(TARGET),target=$(TARGET)) $(if $(LIMIT),limit=$(LIMIT)) $(if $(KB),kb=$(KB)) $(EXTRA)"
	@echo "PID $$(cat $(PIDFILE)), log $(LOG)"
	@sleep 20; \
	if kill -0 $$(cat $(PIDFILE)) 2>/dev/null; then \
	  echo "Running. Check progress with: make status"; \
	else \
	  echo "The training stopped right away. Last lines of the log:"; tail -n 30 $(LOG); exit 1; fi

status:
	@running=$$(pgrep -af '$(TRAIN_RE)'); \
	if [ -n "$$running" ]; then echo "Running:"; echo "$$running"; else echo "No training running."; fi
	@if [ -e $(LOGDIR)/latest.log ]; then \
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
