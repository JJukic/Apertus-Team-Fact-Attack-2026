"""Retry only failed baseline cases with a documented longer SDK timeout.

This transport overlay does not alter prompts, model parameters, source files
or successful predictions. Its source is frozen beside the baseline journal.
All attempts remain in the original request ledger. Forward the baseline
runner's arguments after --request-timeout-seconds.
"""

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import threading
from uuid import uuid4

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--request-timeout-seconds', type=float, default=600)
    parser.add_argument('--source-root', type=Path)
    parser.add_argument('--stream', action='store_true')
    parser.add_argument('--max-in-flight', type=int)
    args, forwarded = parser.parse_known_args()
    if args.request_timeout_seconds <= 0 or '--output' not in forwarded:
        raise ValueError('Positive timeout and baseline --output are required')
    if args.max_in_flight is not None and args.max_in_flight < 1:
        raise ValueError('max-in-flight must be positive')
    output = Path(forwarded[forwarded.index('--output') + 1]).resolve()
    frozen = json.loads((output / 'run.json').read_text()) if (output / 'run.json').exists() else None
    receipt_dir = output / 'transport_overlays'
    receipt_dir.mkdir(parents=True, exist_ok=True)
    identifier = uuid4().hex
    raw = Path(__file__).read_bytes()
    (receipt_dir / f'{identifier}.py').write_bytes(raw)
    receipt = {'baseline_signature': frozen['signature'] if frozen else None,
               'overlay_sha256': hashlib.sha256(raw).hexdigest(),
               'request_timeout_seconds': args.request_timeout_seconds,
               'model_parameter_changes': [], 'retry_scope': 'operational_errors_only' if frozen else 'new_run',
               'stream': args.stream, 'max_in_flight': args.max_in_flight}
    (receipt_dir / f'{identifier}.json').write_text(json.dumps(receipt, indent=2) + '\n')
    # Import the original client only after the runner loads the supplied .env.
    if args.source_root:
        path = args.source_root.resolve() / 'scripts/run_competition_baseline.py'
        spec = importlib.util.spec_from_file_location('isolated_baseline_runner', path)
        baseline = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(baseline)
        receipt['isolated_source_root'] = str(args.source_root.resolve())
    else:
        import scripts.run_competition_baseline as baseline
    from dotenv import load_dotenv
    for i, value in enumerate(forwarded):
        if value == '--env-file':
            load_dotenv(forwarded[i + 1], override=False)
    from src.apertus_client import ApertusClient
    if args.stream:
        from scripts.streaming_transport import assemble_completion
        import scripts.streaming_transport as transport
        transport_raw = Path(transport.__file__).read_bytes()
        receipt['stream_assembler_sha256'] = hashlib.sha256(transport_raw).hexdigest()
        (receipt_dir / f'{identifier}.stream.py').write_bytes(transport_raw)
        initialize = ApertusClient.__init__
        def streaming_initialize(self, *positional, **kwargs):
            initialize(self, *positional, **kwargs)
            create = self.client.chat.completions.create
            def streamed_create(**parameters):
                return assemble_completion(create(**parameters))
            self.client.chat.completions.create = streamed_create
        ApertusClient.__init__ = streaming_initialize
    original = ApertusClient._create_with_retry
    limiter = threading.BoundedSemaphore(args.max_in_flight) if args.max_in_flight else None
    def longer_timeout(self, **kwargs):
        kwargs['timeout'] = args.request_timeout_seconds
        if args.stream:
            kwargs['stream'] = True
            kwargs['stream_options'] = {'include_usage': True}
        # Waiting for this gate is outside the recorded SDK interval; neither
        # local queueing nor backoff is mislabeled as a remote model request.
        if limiter is not None:
            with limiter:
                return original(self, **kwargs)
        return original(self, **kwargs)
    ApertusClient._create_with_retry = longer_timeout
    if frozen and '--retry-errors' not in forwarded:
        forwarded.append('--retry-errors')
    (receipt_dir / f'{identifier}.json').write_text(json.dumps(receipt, indent=2) + '\n')
    sys.argv = [baseline.__file__, *forwarded]
    baseline.main()


if __name__ == '__main__':
    main()
