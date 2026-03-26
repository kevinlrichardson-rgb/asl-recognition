#!/usr/bin/env python
'''Test script to verify that important libraries from the LLM devcontainer are importable.'''

import sys


def test_import(module_name: str, package_name: str | None = None) -> bool:
    '''Attempt to import a module and report success/failure.'''

    display_name = package_name or module_name

    try:
        mod = __import__(module_name)
        version = getattr(mod, '__version__', 'unknown')
        print(f'  [OK] {display_name} ({version})')

        return True

    except ImportError as e:
        print(f'  [FAIL] {display_name}: {e}')

        return False


def main() -> int:
    '''Run all import tests and return exit code.'''

    results: list[bool] = []

    print('\n───LLM Frameworks─────────────────────')
    results.append(test_import('langchain', 'LangChain'))
    results.append(test_import('llama_index', 'LlamaIndex'))
    results.append(test_import('transformers', 'Transformers'))
    results.append(test_import('smolagents', 'smolagents'))

    # vLLM is GPU-only, test but don't fail if unavailable
    print('\n─── GPU-Only (optional) ─────────────────────')
    test_import('vllm', 'vLLM')

    print('\n─── API Clients ─────────────────────')
    results.append(test_import('openai', 'OpenAI'))
    results.append(test_import('anthropic', 'Anthropic'))
    results.append(test_import('ollama', 'Ollama'))

    print('\n─── Vector Store ─────────────────────')
    results.append(test_import('chromadb', 'ChromaDB'))
    results.append(test_import('sentence_transformers', 'sentence-transformers'))

    print('\n─── Tools ─────────────────────')
    results.append(test_import('gradio', 'Gradio'))
    results.append(test_import('accelerate', 'accelerate'))
    results.append(test_import('datasets', 'datasets'))
    results.append(test_import('tiktoken', 'tiktoken'))

    print('\n─── Deep Learning ─────────────────────')
    results.append(test_import('torch', 'PyTorch'))

    # Summary
    passed = sum(results)
    total = len(results)

    print(f'\n─── Summary ─────────────────────')
    print(f'Passed: {passed}/{total}')

    if all(results):
        print('All required imports successful!')
        return 0

    else:
        print('Some imports failed.')
        return 1


if __name__ == '__main__':
    sys.exit(main())
