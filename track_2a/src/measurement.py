"""Local request accounting for competition experiments (no gold data).

Timing approximates proxy instrumentation at SDK call boundaries. It subtracts
the union of request intervals, never the sum of overlapping request latencies.
"""

from contextlib import contextmanager
from contextvars import ContextVar
import hashlib
import json
from pathlib import Path
import threading
import time


def interval_union_seconds(intervals, start, end):
    """Length of clipped, merged in-flight intervals in [start, end]."""
    clipped = sorted((max(start, a), min(end, b)) for a, b in intervals
                     if b > start and a < end and b > a)
    total, right = 0.0, start
    for left, stop in clipped:
        if stop > right:
            total += stop - max(left, right)
            right = stop
    return total


class RequestRecorder:
    def __init__(self, journal: Path, session: str):
        self.journal, self.session = journal, session
        self.journal.parent.mkdir(parents=True, exist_ok=True)
        self.events = []
        self._lock = threading.Lock()
        self._case = ContextVar('competition_case', default=None)

    @contextmanager
    def case(self, case_id):
        token = self._case.set(case_id)
        try:
            yield
        finally:
            self._case.reset(token)

    def wrap(self, create):
        def measured(**kwargs):
            # User instruction: no client-side response token cap. The original
            # production prompt, context and decision rules stay unchanged.
            body = dict(kwargs)
            body.pop('max_tokens', None)
            body.pop('max_completion_tokens', None)
            if any(k in body for k in ('api_key', 'authorization', 'headers')):
                raise ValueError('Credentials must not occur in the recorded request body')
            event = {'session': self.session, 'case_id': self._case.get(),
                     'request_sha256': hashlib.sha256(json.dumps(body, sort_keys=True,
                                                               ensure_ascii=False).encode()).hexdigest(),
                     'output_token_cap': None, 'start': time.perf_counter()}
            event.update(model=body.get('model'), stream=bool(body.get('stream')),
                         response_format=body.get('response_format'))
            try:
                response = create(**body)
                usage = getattr(response, 'usage', None)
                event['input_tokens'] = getattr(usage, 'prompt_tokens', None)
                event['output_tokens'] = getattr(usage, 'completion_tokens', None)
                event['usage_known'] = all(type(event[k]) is int and event[k] >= 0
                                           for k in ('input_tokens', 'output_tokens'))
                choices = getattr(response, 'choices', None) or []
                event['content'] = choices[0].message.content if choices else None
                event['finish_reason'] = getattr(choices[0], 'finish_reason', None) if choices else None
                event['status'] = 'success'
                return response
            except Exception as error:
                # Never persist exception bodies, headers or a credential value.
                event.update(status='error', error_type=type(error).__name__,
                             cause_type=type(error.__cause__).__name__ if error.__cause__ is not None else None,
                             status_code=getattr(error, 'status_code', None),
                             input_tokens=None, output_tokens=None, usage_known=False)
                raise
            finally:
                event['end'] = time.perf_counter()
                with self._lock:
                    self.events.append(event)
                    with self.journal.open('a', encoding='utf-8') as out:
                        out.write(json.dumps(event, ensure_ascii=False) + '\n')
        return measured

    def processing_seconds(self, start, end):
        with self._lock:
            intervals = [(e['start'], e['end']) for e in self.events]
        return max(0.0, end - start - interval_union_seconds(intervals, start, end))
