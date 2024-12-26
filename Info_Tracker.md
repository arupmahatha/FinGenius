For Claude, a token approximately represents 3.5 English characters, though the exact number can vary depending on the language used.
While not exact, a token is generally considered to be around 3/4ths of a word.

Current pricing for Claude 3 Sonnet is:
        Input tokens: $0.003 per 1K tokens
        Output tokens: $0.015 per 1K tokens

Tracking Tokens (Cost) and Time Usage:
1) For easy questions: "Show me the last 5 days of stock prices?"
        Processing Time: 16.67 seconds // 16.00 seconds another try // 15.20 seconds another try
        Token Usage:
        Prompt Tokens: 195, that cost $0.000585 // 195 another try // 195 another try
        Completion Tokens: 788, that cost $0.01182
        Total Cost: $0.012405
2) For complex questions: "Is the stock worth investing?"
        Processing Time: 151.22 seconds // 142.28 seconds another try // 205.34 seconds another try
        Token Usage:
        Prompt Tokens: 2012, that cost $0.006036  // 2029 another try // 1960 another try
        Completion Tokens: 13449 that cost $0.0201735
        Total Cost: $0.0262095

On an average assuming 100 questions with 60% being easy and 40% being complex, the cost for the day would be:
        $0.012405 * 60 + $0.0262095 * 40 = $0.7443 + $1.04838 = $1.79268


Deliverables:
1) Create a dummy data, database schema is given in the file - Done
2) Add memory and human in the loop to the agent - Done
3) Note of processing time - Done
4) Output of the entire analysis (report), like ChatGPT output - Done
5) No CSV uploading, direct SQL queries - Done
6) Examples of strategical level questions - Done

By Monday

POC: 27th Dec, Host it in Streamlit



# Statergic Level Questions
Based on the provided CSV files which contain financial data including balance sheets, income statements, budgets, and forecasts, here are some strategic-level analytical questions that would be valuable for a financial analyst:

1. **Strategic Performance Analysis Question:**
"How does the EBITDA margin trend compare between AC Wailea and Surfrider Malibu properties over the past 12 months, and what are the key drivers behind any significant variances from budgeted figures?"

This question would help analyze:
- Operational efficiency
- Property-wise performance comparison
- Budget accuracy and variance analysis
- Trend identification in profitability

2. **Revenue Optimization Question:**
"What is the correlation between occupancy rates and average daily rates (ADR) for both properties, and how does this relationship impact the overall RevPAR performance compared to budgeted targets?"

This analysis would reveal:
- Pricing strategy effectiveness
- Revenue management opportunities
- Market positioning
- Demand patterns

3. **Cost Structure Analysis Question:**
"How have the operating expense ratios (particularly in rooms and F&B departments) evolved over the past year for both properties, and what impact have these changes had on the gross operating profit margins compared to industry benchmarks?"

This would help understand:
- Cost control effectiveness
- Departmental efficiency
- Margin management
- Operational scalability

These questions leverage data from multiple tables in your database:
- `final_income_sheet_new_seq.csv` for EBITDA and departmental performance
- `final_budget_sheet.csv` for occupancy, ADR, and RevPAR targets
- `final_forecast_sheet.csv` for forward-looking metrics
- `final_income_sheet_tb_new.csv` for detailed expense analysis

Would you like me to elaborate on how to structure SQL queries for any of these analytical questions?


Input tokens when passed each time (Before): 1,917, 2,229 // 
Output tokens when passed once at the beginning (After): 1,590 // 2,836,3,992 // // 

Time(Before): 237.16919827461243, 319.788419008255 // 782.089762210846