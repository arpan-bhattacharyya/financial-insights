
import webbrowser
import csv
import pandas as pd
import yfinance as yf
import numpy as np
import matplotlib.dates as mdate
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import plotly.graph_objects as go




# Base class for handling financial data
class FinancialData:
    def __init__(self, years=0):
        self.years = years
        self.data = []

    def input_into_file(self):



        """
        Basically enters data you give to a csv file called income_statements.csv 
        which comes with the folder when you clone it.
        """
        
        headers = ["Year", "Total Revenue", "Cost of Revenue", "EBIT", "Interest Expense", "Net Income",
                   "Current Assets", "Current Liabilities", "Total Assets", "Total Liabilities", "Total Equity",
                   "Shares Outstanding", "Market Price Per Share", "Dividends Per Share"]
        self.data.append(headers)

        # Collect financial data for each year
        for _ in range(self.years):
            try:
                # Collect inputs for the new fields
                year = input("Enter the year: ")
                total_revenue = float(input("Enter total revenue: "))
                cost_of_revenue = float(input("Enter cost of revenue: "))
                ebit = float(input("Enter EBIT: "))
                interest_expense = float(input("Enter interest expense: "))
                net_income = float(input("Enter net income: "))
                current_assets = float(input("Enter current assets: "))
                current_liabilities = float(input("Enter current liabilities: "))
                total_assets = float(input("Enter total assets: "))
                total_liabilities = float(input("Enter total liabilities: "))
                total_equity = float(input("Enter total equity: "))
                shares_outstanding = float(input("Enter shares outstanding: "))
                market_price_per_share = float(input("Enter market price per share: "))
                dividends_per_share = float(input("Enter dividends per share: "))

                # Append the collected data for the current year
                self.data.append([year, total_revenue, cost_of_revenue, ebit, interest_expense, net_income,
                                  current_assets, current_liabilities, total_assets, total_liabilities, total_equity,
                                  shares_outstanding, market_price_per_share, dividends_per_share])
            except ValueError:
                print("Invalid input. Please enter numeric values.")
                return []  # Exit function if invalid input is encountered

        return self.data

    def save_to_csv(self, filename="income_statements.csv"):



        """
        Saves the data inputed to a csv file name income_statements.csv
        """


        try:
            with open(filename, mode='w', newline="") as file:
                writer = csv.writer(file)
                writer.writerows(self.data)
            print(f"Data has been saved to {filename}")
        except Exception as e:
            print(f"An error occurred while saving to CSV: {e}")

class YahooFinanceData(FinancialData):
    def __init__(self, ticker_symbol):
        super().__init__()
        self.ticker_symbol = ticker_symbol

    def fetch_data_from_yfinance(self):



        """
        So I am using the yfinance API for extracting financial data 
        of listed companies any stock market of the world or atleast 
        which are in the yahoo finance website.
        """


        
        try:
            a = int(input("For how many years of data do you want: "))
            ticker = yf.Ticker(self.ticker_symbol)

            # Fetching financial data
            financials = ticker.financials.transpose().iloc[-a:]
            balance_sheet = ticker.balance_sheet.transpose().iloc[-a:]

            if financials.empty or balance_sheet.empty:
                print(f"No financial data available for the ticker symbol: {self.ticker_symbol}")
                return None

            combined_data_dict = {
                "Year": financials.index.year
            }

            financial_fields = [
                'Total Revenue', 'Cost of Revenue', 'EBIT',
                'Interest Expense', 'Net Income'
            ]
            balance_sheet_fields = [
                'Current Assets', 'Current Liabilities', 'Total Assets', 
                'Total Liabilities', 'Total Equity'
            ]

            # Collecting the financial data
            for field in financial_fields:
                if field in financials.columns:
                    combined_data_dict[field] = financials[field].values
                else:
                    print(f"\nWarning: {field} is missing for {self.ticker_symbol}.")
                    combined_data_dict[field] = [np.nan] * len(financials)

            # Collecting the balance sheet data
            for field in balance_sheet_fields:
                if field in balance_sheet.columns:
                    combined_data_dict[field] = balance_sheet[field].values
                else:
                    print(f"Warning: {field} is missing for {self.ticker_symbol}.\n")
                    combined_data_dict[field] = [np.nan] * len(balance_sheet)

            # Getting some additional data
            shares_outstanding = ticker.info.get('sharesOutstanding', 0)
            market_price_per_share = ticker.history(period="1d")['Close'].iloc[-1]
            dividends_per_share = ticker.info.get('dividendRate', 0)  # Dividends are typically annualized

            # Addition of additional data in combined_data_dict
            num_years = len(financials.index.year)
            combined_data_dict['Shares Outstanding'] = [shares_outstanding] * num_years
            combined_data_dict['Market Price Per Share'] = [market_price_per_share] * num_years
            combined_data_dict['Dividends Per Share'] = [dividends_per_share] * num_years

            # Checking the length of arrays
            lengths = {field: len(arr) for field, arr in combined_data_dict.items()}
            min_length = min(lengths.values())
            
            # Minimizing the array to a minimum length
            for field in combined_data_dict.keys():
                combined_data_dict[field] = combined_data_dict[field][:min_length]

            combined_data = pd.DataFrame(combined_data_dict)
            combined_data.fillna(0, inplace=True)
            combined_data.to_csv("income_statements_yfinance.csv", index=False)
            print(f"Data for {self.ticker_symbol} has been saved to income_statements_yfinance.csv")

            return "income_statements_yfinance.csv"

        except Exception as e:
            print(f"An error occurred while fetching data for {self.ticker_symbol}: {e}")
            return None

    def fetch_stock_prices(self, period="5y", interval="1d"):

        """
        Fetches the historical stock price data for a given ticker symbol from yahoo finance
        """
        try:
            ticker = yf.Ticker(self.ticker_symbol)
            historical_data = ticker.history(period=period, interval=interval)
            
            if historical_data.empty:
                print(f"No price data available for {self.ticker_symbol}")
                return None
            
            historical_data['Date'] = historical_data.index
            historical_data.reset_index(drop=True, inplace=True)
            return historical_data
        except Exception as e:
            print(f"Error fetching stock prices for {self.ticker_symbol}: {e}")
            return None

    def apply_linear_regression(self, data):
        """
        Applies linear regression to the data
        """
        try:
            # Prepare data for regression
            data['Date'] = pd.to_datetime(data['Date'])
            data['Date_ordinal'] = data['Date'].map(mdate.date2num)  # Convert date to numerical value
            X = data[['Date_ordinal']].values
            y = data['Close'].values

            # Split the data into training and testing sets
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

            # Train the linear regression model
            model = LinearRegression()
            model.fit(X_train, y_train)

            # Make predictions
            y_pred_train = model.predict(X_train)
            y_pred_test = model.predict(X_test)

            # Calculate metrics
            mse = np.mean((y_test - y_pred_test) ** 2)
            r2 = r2_score(y_test, y_pred_test)

            print(f"Mean Squared Error: {mse}")
            print(f"R^2 Score: {r2}")

            

            fig = go.Figure()
            fig.add_trace(go.Scatter(x=data['Date'], y=data['Close'], mode='lines', name='Actual Stock Price', line=dict(color='blue')))
            fig.add_trace(go.Scatter(x=data['Date'][:len(y_pred_train)], y=y_pred_train, mode='lines', name='Predicted Price (Train)', line=dict(color='orange')))
            fig.add_trace(go.Scatter(x=data['Date'][len(y_pred_train):], y=y_pred_test, mode='lines', name='Predicted Price (Test)', line=dict(color='green')))
            fig.update_layout(title=f'{self.ticker_symbol} Stock Price Prediction using Linear Regression',
                          xaxis_title='Date',
                          yaxis_title='Stock Price',
                          legend=dict(x=0, y=1))
            fig.show()

        except Exception as e:
            print(f"Error applying linear regression: {e}")

# Derived class for analyzing financial ratios
class RatioAnalysis(FinancialData):
    def __init__(self, csv_file):
        super().__init__()
        self.csv_file = csv_file

    def perform_ratio_analysis(self):

        """
        Performs ratio analysis for the data in the csv file given
        """

       

        
        try:
            data = pd.read_csv(self.csv_file)
        except FileNotFoundError:
            print(f"Error: The file {self.csv_file} was not found.")
        except pd.errors.EmptyDataError:
            print(f"Error: The file {self.csv_file} is empty.")
        except Exception as e:
            print(f"An unexpected error occurred: {e}")

        

        # Calculate financial ratios
        data['Debt_to_Equity_Ratio'] = data['Total Liabilities'] / data['Total Equity'].replace(0, np.nan)
        data['EPS'] = data['Net Income'] / data['Shares Outstanding'].replace(0, np.nan)
        data['PE'] = data['Market Price Per Share'] / data['EPS'].replace(0, np.nan)
        data['ROE'] = data['Net Income'] / data['Total Equity'].replace(0, np.nan)
        data['Current_Ratio'] = data['Current Assets'] / data['Current Liabilities'].replace(0, np.nan)
        data['Dividend_Yield'] = data['Dividends Per Share'] / data['Market Price Per Share'].replace(0, np.nan)
        data['ROI'] = (data['Net Income'] / data['Total Assets']).replace(0, np.nan)

        print("\nCalculated Ratios:")
        print(data[['Year', 'Debt_to_Equity_Ratio', 'EPS', 'PE']])

        print("""\nDisclaimer: The analysis is based on historical averages over several years
                and may not reflect current market conditions, as the stock market is highly dynamic. 
                Past performance is not indicative of future results.\n""")

        for index, row in data.iterrows():
            analysis = []
            if 1 < row['Debt_to_Equity_Ratio'] <1.5 :
                analysis.append("Good Debt-to-Equity Ratio")
            else:
                analysis.append("Bad Debt-to-Equity Ratio")
                
            if row['EPS']>-1:
                analysis.append("Good Earnings Per Share")
            else:
                analysis.append("Bad Earnings Per Share")

            if row['PE'] < 20:
                analysis.append("Good Price-to-Earnings Ratio")
            else:
                analysis.append("Bad Price-to-Earnings Ratio")
            
            if row['ROE'] > 15:
                analysis.append("Good Return on Equity")
            else:
                analysis.append("Bad Return on Equity")
            
            if 1.5 < row['Current_Ratio'] < 3.0:
                analysis.append("Good Current Ratio")
            else:
                analysis.append("Bad Current Ratio")
            
            if 2 < row['Dividend_Yield'] < 5:
                analysis.append("Good Dividend Yield")
            else:
                analysis.append("Bad Dividend Yield")
            
            if row['ROI'] > 10:
                analysis.append("Good ROI")
            else:
                analysis.append("Bad ROI")

            
            
            print(f"Year {row['Year']} {'; '.join(analysis)}")

        

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=data['Year'], y=data['Debt_to_Equity_Ratio'], mode='lines+markers', name='Debt to Equity Ratio', line=dict(color='blue')))
        fig.add_trace(go.Scatter(x=data['Year'], y=data['EPS'], mode='lines+markers', name='EPS', line=dict(color='orange')))
        fig.add_trace(go.Scatter(x=data['Year'], y=data['PE'], mode='lines+markers', name='PE', line=dict(color='green')))
        fig.add_trace(go.Scatter(x=data['Year'], y=data['ROE'], mode='lines+markers', name='ROE', line=dict(color='red')))
        fig.add_trace(go.Scatter(x=data['Year'], y=data['Current_Ratio'], mode='lines+markers', name='Current Ratio', line=dict(color='purple')))
        fig.add_trace(go.Scatter(x=data['Year'], y=data['Dividend_Yield'], mode='lines+markers', name='Dividend Yield', line=dict(color='cyan')))
        fig.add_trace(go.Scatter(x=data['Year'], y=data['ROI'], mode='lines+markers', name='ROI', line=dict(color='magenta')))

        fig.update_layout(title='Financial Ratios Over Years',
                          xaxis_title='Year',
                          yaxis_title='Ratio',
                          legend=dict(x=0, y=1))
        fig.show()


        
# Main function for user interaction
def main():
    """
    Main function for the financial data analysis and stock prediction program that provides users with different options
    to analyze financial data. Users can input data via a CSV file, manually, or by providing a stock ticker symbol.
    """
    print("Welcome to the Financial data analysis and stock prediction program!")
    while True:
        try:
            choice = int(input("1. By adding the financial data's CSV in the folder\n"
                               "2. By manually adding the numbers\n"
                               "3. By giving the ticker symbol of the company\n"
                               "4. Perform stock price prediction using linear regression\n"
                               "5. Exit\n"
                               "Type the serial number: "))

            if choice == 1:
                try:
                    filename = input("Enter the CSV file name: ")
                    analysis = RatioAnalysis(filename)
                    analysis.perform_ratio_analysis()
                except FileNotFoundError:
                     print(f"Error: The file '{filename}' was not found. Please check the file name and try again.")
                except pd.errors.EmptyDataError:
                    print(f"Error: The file '{filename}' is empty or cannot be read properly. Please check the file content.")
                except Exception as e:
                    print(f"An unexpected error occurred: {e}")

            elif choice == 2:
                try:
                    years = int(input("How many years of data? "))
                    manual_input = FinancialData(years)
                    data = manual_input.input_into_file()
                    if data:
                          manual_input.save_to_csv()
                          analysis = RatioAnalysis("income_statements.csv")
                          analysis.perform_ratio_analysis()
                except Exception as e:
                    print(print(f"An unexpected error occurred: {e}"))

            elif choice == 3:
                def tickering():
                    ticker = input("Enter the stock ticker symbol: ")
                    yahoo_data = YahooFinanceData(ticker)
                    historical_data = yahoo_data.fetch_data_from_yfinance()  

                    if historical_data is not None:
                        analysis = RatioAnalysis("income_statements_yfinance.csv")
                        analysis.perform_ratio_analysis()
                try:
                    a = int(input("Do you need help finding the ticker symbol \n"
                               "1. Yes \n"
                               "2. No \n"
                               "Type a serial number: "))
                    if a == 1:
                        print("We are exiting the program to your default web browser")
                        url = "https://finance.yahoo.com/lookup/"
                        webbrowser.open(url)
                        print("Returning to ticker entry after opening browser...")
                        tickering()
                    elif a == 2:
                         tickering()
                except Exception as e:
                    print(print(f"An unexpected error occurred: {e}"))
                    
                
                    
                          
            
                

            elif choice == 4:
                try:
                    ticker = input("Enter the stock ticker symbol: ")
                    period = input("For how many periods i.e  '1y', '2y', '5y', '10y', 'ytd', 'max': ")
                    yahoo_data = YahooFinanceData(ticker)
                    historical_data = yahoo_data.fetch_stock_prices(period=period)
                    if historical_data is not None:
                        yahoo_data.apply_linear_regression(historical_data)
                except Exception as e:
                    print(print(f"An unexpected error occurred: {e}"))
                

            elif choice == 5:
                print("Exiting program.")
                break

            else:
                print("Invalid option. Please try again.")
        except ValueError:
            print("Invalid input. Please enter a valid number.")


main()
