import pandas as pd
from pulp import *

N = 50 # number of stocks
T = 48 # number of periods/months (historical data)
H = -0.15 # threshold
B = 100_000 # budget

# Test the portfolio built by the model
def test(df, result, weights):
    # Return from 1/1/2023 to 1/9/2023 (8 months)
    future = df.iloc[:,4+T::8].pct_change(axis='columns') 
    sum = 0 
    for i in result:
        sum += (1+future.iloc[i,1:]) * weights[i] * B
    total_return = (sum - B)/B
    return total_return.iloc[0]
    
# Sectors
Info_Tech = []
Consumer_Staples = []
Consumer_Discretionary = []
Energy = []
Financials = []
Health_Care = []
Industrials = []
Materials = []
Real_Estate = []
Communication_Services = []
Utilities = []

# Split the stocks into sectors
def sectors(df):
    for i in range(N):
        sector = df.iloc[i,3]
        if (sector == "Info Tech"):
            Info_Tech.append(i)
        elif (sector == "Consumer Staples"):
            Consumer_Staples.append(i)
        elif (sector == "Consumer Discretionary"):
            Consumer_Discretionary.append(i)
        elif (sector == "Energy"):
            Energy.append(i)
        elif (sector == "Financials"):
            Financials.append(i)
        elif (sector == "Health Care"):
            Health_Care.append(i)
        elif (sector == "Industrials"):
            Industrials.append(i)
        elif (sector == "Materials"):
            Materials.append(i)
        elif (sector == "Real Estate"):
            Real_Estate.append(i)
        elif (sector == "Communication Services"):
            Communication_Services.append(i)
        elif (sector == "Utilities"):
            Utilities.append(i)

    return [Info_Tech, Consumer_Staples, Consumer_Discretionary, Energy, Financials, Health_Care, \
            Industrials, Materials, Real_Estate, Communication_Services, Utilities]
            

if __name__ == "__main__":
    
    # Import data from excel file into a dataframe
    df = pd.read_excel("data.xlsx")
    
    # Get the return of each period/month for every stock (4 years = 48 months)
    y = df.iloc[:,4:T+5].pct_change(axis='columns') #y(it) = (q(it) - q(it-1))/q(it-1)
    
    # Get the mean of the returns for every stock 
    mean = y.mean(axis=1) #y(i) = (y(i1) + y(i2) + ... + y(iT))/T
    
    sectors(df)

    # Create the LP problem
    prob = LpProblem("Portfolio_Optimazation", LpMaximize)

    # Decision variables
    W = []
    for i in range(N):
        W.append(LpVariable(name='W{}'.format(i), lowBound=0, upBound=1))

    # Objective function
    prob += lpSum([mean[i]*W[i] for i in range(N)]), "Average Return"

    # Constraints

    # Return of each period examined greater than Threshold H
    for t in range(1, T):
        prob += lpSum([W[i]*y.iloc[i,t] for i in range(N)]) >= H, "Return of {} period".format(t) 
    
    prob += lpSum([W[i] for i in range(N)]) <= 1, "Sum of weights less than 1" 
    
    # Weight of each stock less than 0.15 (No stock can account for more than 15% of the portfolio)
    for i in range(N):
        prob += W[i] <=0.15, "Weight {} less than 0.15".format(i) 
    
    # Sector constraints (No sector can account for more than each limit of the portfolio)
    prob += lpSum([W[i] for i in Info_Tech]) <= 0.4, "Info_Tech"
    prob += lpSum([W[i] for i in Consumer_Staples]) <= 0.2, "Consumer_Staples"
    prob += lpSum([W[i] for i in Consumer_Discretionary]) <= 0.3, "Consumer_Discretionary"
    prob += lpSum([W[i] for i in Energy]) <= 0.15, "Energy"
    prob += lpSum([W[i] for i in Financials]) <= 0.3, "Financials"
    prob += lpSum([W[i] for i in Health_Care]) <= 0.3, "Health_Care"
    prob += lpSum([W[i] for i in Industrials]) <= 0.2, "Industrials"
    prob += lpSum([W[i] for i in Materials]) <= 0.1, "Materials"
    prob += lpSum([W[i] for i in Real_Estate]) <= 0.1, "Real_Estate"
    prob += lpSum([W[i] for i in Communication_Services]) <= 0.25, "Communication_Services"
    prob += lpSum([W[i] for i in Utilities]) <= 0.1, "Utilities"

    # Solve the problem
    prob.solve()
    print("Status:", LpStatus[prob.status])

    # Print the optimal value of the objective function
    print("Expected monthly portfolio return = ", value(prob.objective))

    # Export results 
    export = pd.DataFrame(columns=[0,1,2,3], dtype=object)
    j=1
    for i in range(N):
        if (W[i].varValue != 0):
            export.loc[j, 0] = df.iloc[i, 0]
            export.loc[j, 1] = df.iloc[i, 1]
            export.loc[j, 2] = df.iloc[i, 3]
            export.loc[j, 3] = W[i].varValue * B
            j+=1
    export.to_excel("export.xlsx", index=False, header=["Stock", "Symbol", "Sector", "Allocation"])

    result =[]
    # Print the results
    for i in range(N):
        if (W[i].varValue != 0):
            result.append(i)
    
    for i in result:    
        print("Weight of {} = {} or {}$".format(df.iloc[i, 0], W[i].varValue, W[i].varValue * B))

    # Print the sum of the weights
    print("Sum of weights = ", value(sum([v.varValue for v in prob.variables()])))
    
    print("\n")
    # Print the total return on portfolio 
    print("Total return on portfolio (in 2023): " , test(df, result, [W[i].varValue for i in range(N)]))
    print("\n")