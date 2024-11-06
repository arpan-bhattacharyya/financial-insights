

# Financial Data Analysis and Stock Prediction Program

This program allows users to analyze financial data, perform ratio analysis, and predict stock prices using linear regression. The program supports multiple ways to input data and analyze stocks, including manual data entry, reading data from a CSV file, and fetching data from Yahoo Finance using a ticker symbol.

## Features

- **Manual Data Entry**: Users can input financial data for multiple years manually, including details such as revenue, net income, assets, and more.
- **CSV Import**: Users can upload a CSV file containing financial data for ratio analysis.
- **Yahoo Finance Integration**: Fetch financial and stock price data directly from Yahoo Finance using a ticker symbol.
- **Stock Price Prediction**: Uses linear regression to predict future stock prices based on historical data.
- **Ratio Analysis**: Calculates and analyzes various financial ratios such as Debt-to-Equity, EPS, P/E, ROE, Current Ratio, Dividend Yield, and ROI over multiple years.

## Requirements

Ensure the following libraries are installed:

- `pandas`
- `yfinance`
- `numpy`
- `matplotlib`
- `plotly`
- `sklearn`
- `csv`

You can install the required libraries using `pip`:

```bash
pip install pandas yfinance numpy matplotlib plotly scikit-learn
```

## Usage

### 1. **By Adding the Financial Data's CSV in the Folder**
   - Place a CSV file with financial data in the folder.
   - The CSV file must contain columns like `Year`, `Total Revenue`, `Cost of Revenue`, etc.
   - Select the option to import the CSV file, and the program will perform a ratio analysis based on the data in the file.

### 2. **By Manually Adding the Numbers**
   - Enter the number of years of financial data.
   - The program will prompt you to input financial details for each year.
   - After entering the data, it will be saved to a CSV file for future analysis.

### 3. **By Providing a Stock Ticker Symbol**
   - Input the ticker symbol of a company (e.g., `AAPL` for Apple).
   - The program will fetch the company's financial and stock price data from Yahoo Finance.
   - The program will also allow you to open Yahoo Finance in your default browser to search for a ticker symbol if you're unsure.

### 4. **Perform Stock Price Prediction Using Linear Regression**
   - Enter a stock ticker symbol and select the period (e.g., '1y', '2y', '5y', etc.).
   - The program fetches historical stock prices for the given period and applies linear regression to predict future stock prices.

### 5. **Exit**
   - Exit the program.

## Functions

### `FinancialData`
A base class for handling financial data, including input, saving to CSV, and basic financial data management.

### `YahooFinanceData`
A derived class that fetches financial data and stock prices using Yahoo Finance's API. It also applies linear regression for stock price prediction.

### `RatioAnalysis`
A derived class that performs financial ratio analysis on the given CSV file, calculating ratios like Debt-to-Equity, EPS, P/E, ROE, and others. It also generates visualizations of these ratios over time.

## Example Output

For example, when performing ratio analysis, the program might output:

```
Calculated Ratios:
    Year  Debt_to_Equity_Ratio   EPS  PE
0   2020                     1.2  5.6  18
1   2021                     1.3  6.1  20

Disclaimer: The analysis is based on historical averages over several years and may not reflect current market conditions. Past performance is not indicative of future results.

Year 2020 Good Debt-to-Equity Ratio; Good Earnings Per Share; Good Price-to-Earnings Ratio
Year 2021 Good Debt-to-Equity Ratio; Good Earnings Per Share; Bad Price-to-Earnings Ratio
```

Additionally, the program generates interactive plots using Plotly to visualize the financial ratios over time.

## Disclaimer

The analysis and predictions are based on historical data and linear regression models, which may not accurately predict future stock prices or financial performance. Always conduct your own research and consult with financial experts before making investment decisions.

## Contributing

Feel free to fork this project, submit issues, and send pull requests. Contributions are welcome!

