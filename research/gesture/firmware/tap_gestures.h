/*
 * tap_gestures.h - single tap / double tap decision on top of the TAP_SWIPE onset score.
 *
 * Call tg_update() once per model run with the Q20 TAP_SWIPE score (0..20) and a millisecond clock.
 * It returns TG_TAP, TG_DOUBLE_TAP or TG_NONE.
 *
 *   - A tap starts on a rising edge of the score to >= TG_FIRE_Q20 (0.85, the trained threshold).
 *   - The 2nd tap of a double needs only TG_SECOND_Q20 (0.60), but only inside the double-tap window
 *     (TG_MIN_GAP_MS..TG_WINDOW_MS after the 1st) and only after the score fell below TG_DIP_Q20 (0.20)
 *     since the 1st tap, so the release of one tap cannot count as a 2nd tap. Second taps score lower than
 *     first taps (device logs: 0.55-0.80 when missed at 0.85); single taps showed no bump after the dip.
 *     Nothing changes outside the window, so single-tap false events stay as they are.
 *   - TG_REPORT_TAP_IMMEDIATELY 0 (default): TAP and DOUBLE_TAP are exclusive. A single tap is reported
 *     when the window closes (TG_WINDOW_MS after its onset); a double tap is reported at the 2nd tap.
 *     TG_REPORT_TAP_IMMEDIATELY 1: TAP at once on every 1st tap, DOUBLE_TAP follows (the app undoes the TAP).
 *   - After a double tap, onsets are ignored for TG_LOCKOUT_MS, so a triple tap is not double + single.
 *   - TG_CHAIN_TAPS 1 (needs TG_REPORT_TAP_IMMEDIATELY 1; HID clicks): no lockout, every accepted tap re-arms
 *     the window, so in a fast series (4-5 taps/s) every tap after the first is TG_DOUBLE_TAP with the
 *     relaxed threshold. Send one click per TG_TAP and per TG_DOUBLE_TAP; the host forms double clicks.
 *   - tg_cancel() drops a pending (not yet reported) tap, e.g. when a swipe or long press is recognised.
 *
 * Integer only, 12 bytes of state, about 67 Cortex-M4 instructions, no calls. The thresholds and windows were chosen on
 * held-out replay (DoubleTaps / single-tap / swipe / movement recordings) and checked on device logs, see
 * research/gesture/TAP_GESTURES.md. Host twin: research/gesture/tap_gestures.py (bit-exact, tested).
 */
#ifndef TAP_GESTURES_H
#define TAP_GESTURES_H

#include <stdint.h>

#ifndef TG_FIRE_Q20
    #define TG_FIRE_Q20 (17u)              /* 0.85: tap onset (rising edge), unchanged */
#endif
#ifndef TG_SECOND_Q20
    #define TG_SECOND_Q20 (12u)            /* 0.60: 2nd tap of a double, inside the window, after a dip */
#endif
#ifndef TG_DIP_Q20
    #define TG_DIP_Q20 (4u)                /* 0.20: the score must fall below this between the two taps */
#endif
#ifndef TG_MIN_GAP_MS
    #define TG_MIN_GAP_MS (150u)           /* refractory: onsets closer than this are the same tap */
#endif
#ifndef TG_WINDOW_MS
    #define TG_WINDOW_MS (400u)            /* 2nd tap must start within this time of the 1st */
#endif
#ifndef TG_LOCKOUT_MS
    #define TG_LOCKOUT_MS (300u)           /* after a double tap, ignore onsets this long */
#endif
#ifndef TG_REPORT_TAP_IMMEDIATELY
    #define TG_REPORT_TAP_IMMEDIATELY (0)
#endif
#ifndef TG_CHAIN_TAPS
    #define TG_CHAIN_TAPS (0)
#endif
#if (TG_CHAIN_TAPS != 0) && (TG_REPORT_TAP_IMMEDIATELY == 0)
    #error "TG_CHAIN_TAPS needs TG_REPORT_TAP_IMMEDIATELY 1"
#endif

typedef enum
{
    TG_NONE = 0,
    TG_TAP,
    TG_DOUBLE_TAP,
} tg_event_t;

enum
{
    TG_IDLE = 0,
    TG_PENDING,                            /* one tap seen, waiting for a 2nd */
    TG_LOCKOUT,                            /* just reported a double tap */
};

typedef struct
{
    uint32_t t_first;                      /* ms, onset of the pending tap */
    uint32_t t_last;                       /* ms, last accepted onset */
    uint8_t  state;
    uint8_t  prev_q;
    uint8_t  dipped;
    uint8_t  reserved;
} tg_state_t;

/* Call when the sensor stream (re)starts or the model is gated off. */
static inline void tg_reset(tg_state_t* s, uint32_t now_ms)
{
    s->state   = TG_IDLE;
    s->prev_q  = 0u;
    s->dipped  = 0u;
    s->t_first = now_ms;
    s->t_last  = now_ms - TG_LOCKOUT_MS - TG_MIN_GAP_MS; /* no refractory right after a reset */
}

/* Drop a pending tap that was not reported yet (deferred mode), e.g. it turned out to be a swipe. */
static inline void tg_cancel(tg_state_t* s)
{
    if (s->state == TG_PENDING)
    {
        s->state = TG_IDLE;
    }
}

static inline tg_event_t tg_update(tg_state_t* s, uint8_t q, uint32_t now_ms)
{
    tg_event_t    ev     = TG_NONE;
    const uint8_t rising = (uint8_t) ((q >= TG_FIRE_Q20) && (s->prev_q < TG_FIRE_Q20));
    s->prev_q            = q;

    if (s->state == TG_PENDING)
    {
        const uint32_t dt = now_ms - s->t_first;
        if (q < TG_DIP_Q20)
        {
            s->dipped = 1u;
        }
        if ((dt >= TG_MIN_GAP_MS) && (dt <= TG_WINDOW_MS) && (s->dipped != 0u) && (q >= TG_SECOND_Q20))
        {
#if (TG_CHAIN_TAPS != 0)
            s->t_first = now_ms;           /* stay pending: the next tap of a fast series counts too */
            s->dipped  = 0u;
#else
            s->state = TG_LOCKOUT;
#endif
            s->t_last = now_ms;
            return TG_DOUBLE_TAP;
        }
        if (dt <= TG_WINDOW_MS)
        {
            return TG_NONE;
        }
        s->state = TG_IDLE;                /* the window closed: it was a single tap */
#if (TG_REPORT_TAP_IMMEDIATELY == 0)
        ev = TG_TAP;
#endif
    }
    if (s->state == TG_LOCKOUT)
    {
        if ((now_ms - s->t_last) < TG_LOCKOUT_MS)
        {
            return TG_NONE;
        }
        s->state = TG_IDLE;
    }
    if ((rising != 0u) && ((now_ms - s->t_last) >= TG_MIN_GAP_MS))
    {
        s->state   = TG_PENDING;
        s->t_first = now_ms;
        s->t_last  = now_ms;
        s->dipped  = 0u;
#if (TG_REPORT_TAP_IMMEDIATELY != 0)
        ev = TG_TAP;
#endif
    }
    return ev;
}

#endif /* TAP_GESTURES_H */
