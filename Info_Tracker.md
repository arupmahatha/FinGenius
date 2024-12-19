For Claude, a token approximately represents 3.5 English characters, though the exact number can vary depending on the language used.
While not exact, a token is generally considered to be around 3/4ths of a word.

Current pricing for Claude 3 Sonnet is:
        Input tokens: $0.003 per 1K tokens
        Output tokens: $0.015 per 1K tokens

Tracking Tokens (Cost) and Time Usage:
1) For easy questions: "Show me the last 5 days of stock prices?"
        Processing Time: 16.67 seconds // 16.00 seconds another try
        Token Usage:
        Prompt Tokens: 195, that cost $0.000585
        Completion Tokens: 788, that cost $0.01182
        Total Cost: $0.012405
2) For complex questions: "Is the stock worth investing?"
        Processing Time: 151.22 seconds // 142.28 seconds another try
        Token Usage:
        Prompt Tokens: 2012, that cost $0.006036  // 2029 another try
        Completion Tokens: 13449 that cost $0.0201735
        Total Cost: $0.0262095

On an average assuming 100 questions with 60% being easy and 40% being complex, the cost for the day would be:
        $0.012405 * 60 + $0.0262095 * 40 = $0.7443 + $1.04838 = $1.79268