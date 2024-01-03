import pymysql
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, classification_report, mean_squared_error
from sklearn.preprocessing import StandardScaler

db = pymysql.connect(host='localhost',
 user='root',
 passwd='eecs118finaltermproject',
 db= 'flights')
cur = db.cursor()

def query1(city):
    sql = "SELECT * FROM Flight_leg WHERE Departure_airport_code IN (SELECT Airport_code FROM Airport WHERE City = %s)"
    cur.execute(sql, (city,))
    results = cur.fetchall()
    if not results:
        print("\nNo flights found departing from", city)
    else:
        print(f"\nThe flights departing from {city} are as follows:")
        for row in results:
            flight_number, leg_number, depart_code, depart_time, arrival_airport, arrival_time  = row
            print(f"\nFlight Number: {flight_number}, Leg Number: {leg_number}, Departure Airport: {depart_code}, Departure Time: {depart_time}, Arrival Airport: {arrival_airport}, Arrival Time: {arrival_time}")
    print("\n")

def query2(customer_name):
    sql = "SELECT Flight_number, Seat_number FROM Seat_reservation WHERE Customer_name = %s"
    cur.execute(sql, (customer_name,))
    results = cur.fetchone()
    if not results:
        print("\nNo customer found\n")
    else:
        flight_number, seat_number = results
        print(f"\n{customer_name} is on flight number {flight_number} and is in seat number {seat_number}\n ")

def query3(departure_airport, destination_airport, date):
    sql = """SELECT Leg_instance.Flight_number, MIN(Fare.Amount) as MinFare
            FROM flights.Leg_instance
            JOIN flights.Fare ON Leg_instance.Flight_number = Fare.Flight_number
            WHERE Leg_instance.Departure_airport_code = %s
            AND Leg_instance.Arrival_airport_code = %s
            AND Leg_instance.leg_date = %s
            GROUP BY Leg_instance.Flight_number"""

    cur.execute(sql, (departure_airport, destination_airport, date))
    result = cur.fetchone()
    if not result or result[1] is None:
        print(f"\nNo flights found from {departure_airport} to {destination_airport} on {date}\n")
    else:
        flight_number, min_fare = result
        print(f"\nThe cheapest flight is Flight Number {flight_number}, and the cost is ${min_fare}\n")

def query4(airline_name):
    sql_non_stop_flights = """
    SELECT DISTINCT F1.Flight_number
    FROM flights.Flight F1
    JOIN flights.Leg_instance L1 ON F1.Flight_number = L1.Flight_number
    LEFT JOIN flights.Leg_instance L2 ON F1.Flight_number = L2.Flight_number AND L2.Leg_number <> L1.Leg_number
    WHERE F1.Airline = %s
      AND L2.Flight_number IS NULL;
    """

    cur.execute(sql_non_stop_flights, (airline_name,))
    results = cur.fetchall()

    if not results:
        print(f"\nNo non-stop flights found for airline {airline_name}\n")
    else:
        flight_numbers = [result[0] for result in results]
        print(f"\nThe non-stop flights for airline {airline_name} are: {', '.join(flight_numbers)}\n")

def query5(date):
    sql = """
    SELECT DISTINCT Flight_number
    FROM flights.Leg_instance
    WHERE leg_date = %s;
    """
    cur.execute(sql, (date,))
    results = cur.fetchall()

    if not results:
        print(f"\nNo flights found for the date {date}\n")
    else:
        flight_numbers = [result[0] for result in results]
        print(f"\nThe flights for the date {date} are: {', '.join(flight_numbers)}\n")
    
def query6(N):
    df=pd.read_csv('cereal.csv')
    df.reset_index(drop=True, inplace=True)
    df_sorted = df.sort_values(by='rating', ascending=False)
    
    #df.index=df.index.astype(int)
    print(df_sorted.head(N))
    #top_cereals = df_sorted.reset_index(drop=True).head(N)

    top_cereals=df_sorted.head(N)
    plt.figure(figsize=(12, 6))
    plt.bar(top_cereals['name'], top_cereals['rating'], color='skyblue')
    plt.xlabel('Cereal Name')
    plt.ylabel('Rating')
    plt.title(f'Top {N} Cereals with the Highest Ratings')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()

def query7(x_column, y_column):
    df=pd.read_csv('cereal.csv')
    if x_column not in df.columns or y_column not in df.columns:
        print("\nInvalid choice for values. Try again.\n")
        return
    X=df[[x_column]]
    y=df[y_column]
    model=LinearRegression()
    model.fit(X, y)
    plt.scatter(X, y, label='Data Points')
    plt.plot(X, model.predict(X), color='red', label='Linear Regression')
    plt.title(f'Scatter Plot with Linear Regression: {x_column} vs {y_column}')
    plt.xlabel(x_column)
    plt.ylabel(y_column)
    plt.legend()
    print(f'\nRegression Equation: {y_column} = {model.coef_[0]:.2f} * {x_column} + {model.intercept_:.2f}\n')
    plt.show()

def query8(pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, diabetespedigree, age):
    data=pd.read_csv('diabetes.csv')
    X = data.drop('Outcome', axis=1)
    y = data['Outcome']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    clf = DecisionTreeClassifier(random_state=42)

    clf.fit(X_train, y_train)
    input_data=pd.DataFrame({
        'Pregnancies': [pregnancies],
        'Glucose': [glucose],
        'BloodPressure': [blood_pressure],
        'SkinThickness': [skin_thickness],
        'Insulin': [insulin],
        'BMI': [bmi],
        'DiabetesPedigreeFunction': [diabetespedigree],
        'Age': [age]
    })
    predictions = clf.predict(input_data)


    print("Prediction:")
    for prediction in predictions:
        if prediction == 0:
            print("No Diabetes")
        else:
            print("Has Diabetes")


    print("Decision Tree Visualization:")
    plt.figure(figsize=(20, 10))
    plot_tree(clf, feature_names=input_data.columns.tolist(), class_names=["No Diabetes", "Diabetes"], filled=True, rounded=True)
    plt.show()

def query9():
    df=pd.read_csv('diabetes.csv')
    X = df.drop('Outcome', axis=1)
    y = df['Outcome']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    alpha = 1.0
    ridge = Ridge(alpha=alpha)
    ridge.fit(X_train_scaled, y_train)
    y_pred=ridge.predict(X_test_scaled)
    mse = mean_squared_error(y_test, y_pred)
    print(f'\nMean Squared Error: {mse}')

    print('\nRegression Equation:')
    for feature, coefficient in zip(X.columns, ridge.coef_):
        print(f'{feature}: {coefficient:.4f}')

    print(f'Intercept: {ridge.intercept_}\n')

def query10(seat):
    df=pd.read_csv('flight.csv')
    reservations=df[df['Seat_number']==seat]
    if reservations.empty:
        print("\nThere are no reservations for this seat.\n")
    else: 
        reservations_string=reservations.to_string(index=False)
        print(f"\nReservations for seat {seat}:\n{reservations_string}\n")
        print("\n")

def query11():
    df=pd.read_csv('AdverseFoodEvents.csv')
    industry_counts=df['PRI_FDA Industry Name'].value_counts()
    print("\nCount of occurrences for each unique value in 'PRI_FDA Industry Name':")
    print(industry_counts)
    print("\n")

def main():
    while True:
        print("Query Options:")
        print("1. Find all flights departing from a specific city.")
        print("2. Find flight and seat information for a customer.")
        print("3. Find the cheapest flight given departure location, destination, and date")
        print("4. Find all non-stop flights for an airline.")
        print("5. Find all flights occuring during a single date.")
        print("6. Find the top 'N' cereals with the highest ratings")
        print("7. Find the linear regression equation between two parameters of cereals and plot it i.e. calories vs. fat")
        print("8: Train a Decision Tree classifier for diabetes possibility and make a prediction")
        print("9. Train and test a ridge regression model for the diabetes dataset, where X is all the columns besides outcome and Y is the outcome")
        print("10: Find information for all flight reservations made for a certain seat number")
        print("11. Find the number of occurences for each industry in the Adverse Food Events dataset.")
        print("12: Quit")
        choice = input("Enter the number of the option you would like to choose: ")
        if choice=='1':
            city=input("Please enter city name (for cities w/ spaces, use a '-' instead of a space i.e. 'Los-Angeles'): ")
            query1(city)
        elif choice=='2':
            customer_name = input("Please enter the customer's name: ")
            query2(customer_name)
        elif choice=='3':
            departure_airport=input("Enter the departing city in acronym format i.e. 'SFO': ")
            destination_airport=input("Enter the destination city in acronym format i.e. 'SFO': ")
            date=input("Enter the flight date (yyyy-mm-dd): ")
            query3(departure_airport, destination_airport, date)
        elif choice=='4':
            airline_name = input("Enter the name of the airline (Case sensitive): ")
            query4(airline_name)
        elif choice=='5':
            date=input("Enter the date of the flights you would like shown (yyy-mm-dd): ")
            query5(date)
        elif choice=='6':
            nstr=input("Enter the length N of the list you would like to see: ")
            n=int(nstr)
            query6(n)
        elif choice=='7':
            x_col=input("Enter in the x value you would like to use (ensure lowercase and quantitative data i.e. sugars): ")
            y_col=input("Enter in the y value you would like to use: ")
            query7(x_col, y_col)
        elif choice=='8':
            pregnancies=input("Enter the number of pregnancies of the patient: ")
            glucose=input("Enter the glucose level of the patient: ")
            bloodpressure=input("Enter the blood pressure of the patient: ")
            skinthickness=input("Enter the skin thickness of the patient: ")
            insulin=input("Enter the insulin of the patient: ")
            bmi=input("Enter the bmi of the patient: ")
            diabetespedigree=input("Enter the diabetes pedigree function of the patient: ")
            age=input("Enter the age of the patient: ")
            query8(pregnancies, glucose, bloodpressure, skinthickness, insulin, bmi, diabetespedigree, age)
        elif choice=='9':
            query9()
        elif choice=='10':
            seat=input("Enter in the seat number i.e. '7A': ")
            query10(seat)
        elif choice=='11':
            query11()
        elif choice=='12':
            break
        else: 
            print("\nInvalid choice. Please enter in a valid option number.\n")

if __name__ == "__main__":
    main()

db.close()