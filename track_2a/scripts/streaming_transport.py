"""Assemble a streamed Apertus response into the production client's interface.

Streaming changes HTTP delivery only; it does not impose a token cap or alter
the prompt/decoding options. RequestRecorder must surround the full assembly,
so the complete in-flight interval remains excluded from local processing time.
"""

from types import SimpleNamespace


def assemble_completion(stream):
    content, usage, finish_reason = [], None, None
    with stream:
        for chunk in stream:
            if getattr(chunk, 'usage', None) is not None:
                usage = chunk.usage
            for choice in getattr(chunk, 'choices', None) or []:
                if getattr(choice, 'index', 0) != 0:
                    raise ValueError('Production NLI expects exactly one completion choice')
                delta = choice.delta
                text = getattr(delta, 'content', None)
                if text:
                    content.append(text)
                if getattr(choice, 'finish_reason', None):
                    finish_reason = choice.finish_reason
    if finish_reason is None:
        raise ValueError('Stream ended without a completion finish marker')
    return SimpleNamespace(
        usage=usage,
        choices=[SimpleNamespace(message=SimpleNamespace(content=''.join(content)),
                                 finish_reason=finish_reason, logprobs=None)],
    )
