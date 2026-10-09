#!/usr/bin/env bash
# Wait until Groq's rolling per-day budget can afford a full agent turn.
#
# How it reads the budget without spending anything:
#
#   * A turn makes 3 LLM calls (~4,500 tokens of prompt + schema).
#   * Send one deliberately *large* prompt that is bigger than today's
#     remaining budget. It is always refused, and the 429 body reports
#     "Used N" — which pins the exact consumption. A refused request costs
#     zero tokens, so this probe is free.
#   * The probe prompt must stay larger than $TARGET, otherwise it would
#     start succeeding and consume its own prompt on the way in. It must
#     also stay under Groq's per-request input ceiling (~7k tokens) or it
#     returns 413 instead of 429 and reports no usage at all.
#
# Prints "READY <headroom>" and exits 0 once a turn is affordable.
set -u
KEY=$(grep '^GROQ_API_KEY=' .env | cut -d= -f2)
URL=https://api.groq.com/openai/v1/chat/completions
TARGET=${1:-4500}          # tokens a full turn needs
DEADLINE=$(( $(date +%s) + ${2:-3600} ))

# Probe sizing is load-bearing and must use natural language, not a repeated
# filler character: 30k chars of English ~= 5,570 tokens (measured 5.37
# chars/token), which sits above TARGET — so the probe keeps being refused and
# costs nothing — and below Groq's ~7k per-request input ceiling, where the
# API answers 413 instead of 429 and stops reporting usage at all.
#
# Repeating a single character instead of a sentence is a trap: the tokenizer
# merges runs of identical characters, so the probe is far smaller than its
# byte count suggests and silently starts *succeeding* — spending its whole
# prompt on the way in and burning exactly the budget being waited for.
SENT="The hotel allowance for a domestic trip is one thousand five hundred Egyptian pounds per night, taxi fares are capped at eight hundred pounds per leg, and a team dinner costs two thousand pounds per person under the expense policy. "
# 150 lines ~= 34.8k chars ~= 6,470 tokens (measured 5.38 chars/token): above
# TARGET=5600, still below the ~7k per-request input ceiling.
PROMPT=$(yes "$SENT" | head -n 150 | tr -d '\n')

while [ "$(date +%s)" -lt "$DEADLINE" ]; do
  OUT=$(curl -s -m 60 -w '\n%{http_code}' "$URL" \
    -H "Authorization: Bearer $KEY" \
    -H 'Content-Type: application/json' \
    -H 'User-Agent: Mozilla/5.0' \
    -d "{\"model\":\"qwen/qwen3.8-27b\",\"messages\":[{\"role\":\"user\",\"content\":\"$PROMPT\"}],\"max_tokens\":1}")
  CODE=$(printf '%s' "$OUT" | tail -1)
  BODY=$(printf '%s' "$OUT" | sed '$d')

  if [ "$CODE" = "200" ]; then
    # Only reachable if headroom ever exceeds the probe itself — treat as ready.
    echo "READY: probe succeeded at $(date +%H:%M:%S)"
    exit 0
  fi

  USED=$(printf '%s' "$BODY" | grep -o 'Used [0-9]*' | grep -o '[0-9]*')
  REQD=$(printf '%s' "$BODY" | grep -o 'Requested [0-9]*' | grep -o '[0-9]*')
  if [ -z "${USED:-}" ]; then
    echo "$(date +%H:%M:%S) HTTP $CODE $(printf '%s' "$BODY" | head -c 140)"
  else
    HEAD=$((200000 - USED))
    echo "$(date +%H:%M:%S) headroom=${HEAD} tokens (probe needs ${REQD:-?})"
    if [ "$HEAD" -ge "$TARGET" ]; then
      echo "READY: headroom=${HEAD} >= target=${TARGET} at $(date +%H:%M:%S)"
      exit 0
    fi
  fi
  sleep 30
done
echo "TIMEOUT waiting for quota"
exit 1
