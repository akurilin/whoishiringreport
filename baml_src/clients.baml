// LLM clients for job extraction
// https://docs.boundaryml.com/docs/snippets/clients/overview

// OpenAI models
client<llm> GPT4oMini {
  provider openai
  retry_policy Exponential
  options {
    model "gpt-4o-mini"
    api_key env.OPENAI_API_KEY
  }
}

client<llm> GPT4o {
  provider openai
  retry_policy Exponential
  options {
    model "gpt-4o"
    api_key env.OPENAI_API_KEY
  }
}

// Gemini models
client<llm> Gemini2FlashLite {
  provider google-ai
  retry_policy Exponential
  options {
    model "gemini-2.0-flash-lite"
    api_key env.GEMINI_API_KEY
  }
}

client<llm> Gemini25FlashLite {
  provider google-ai
  retry_policy Exponential
  options {
    model "gemini-2.5-flash-lite"
    api_key env.GEMINI_API_KEY
  }
}

client<llm> Gemini2Flash {
  provider google-ai
  retry_policy Exponential
  options {
    model "gemini-2.0-flash"
    api_key env.GEMINI_API_KEY
  }
}

// Retry policy for all clients
retry_policy Exponential {
  max_retries 2
  strategy {
    type exponential_backoff
    delay_ms 300
    multiplier 1.5
    max_delay_ms 10000
  }
}
