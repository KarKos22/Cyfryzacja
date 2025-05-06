
import streamlit as st
import pandas as pd
import geopandas as gpd
import plotly.express as px
import altair as alt
from urllib.error import URLError
from scipy.stats import linregress
import pymannkendall as mk
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm

# --- Sekcja wyboru strony ---
st.sidebar.title('Menu')
selected_section = st.sidebar.selectbox('Poziom agregacji', ['Liczność firm', 'Mapy','Branże','Korelacje-technologia','Korelacje-informacje','Korelacje-turystyka','Korelacje-budownictwo','Korelacje-przemysł','Korelacje-handel','Regresja liniowa'])

############################REGRESJA###################################################################################################
def regresja_liniowa_section():
    selected_subsection = st.sidebar.selectbox("Wybierz kategorię", [
        "Nakłady", "Pracownicy", "Wartość Nakładów", "Rodzaje Połączeń", "Prędkość Połączeń"
    ])
    if selected_subsection == "Nakłady":
        r_naklady_section()
    elif selected_subsection == "Pracownicy":
        r_pracownicy_section()
    elif selected_subsection == "Wartość Nakładów":
        r_wartosc_nakladow_section()
    elif selected_subsection == "Rodzaje Połączeń":
        r_rodzaje_polaczen_section()
    elif selected_subsection == "Prędkość Połączeń":
        r_predkosc_polaczen_section()
        
def r_naklady_section():
    st.sidebar.header("Nakłady")
    pages = {
        "Nakłady na sprzęt/oprogramowanie": r_nak_sprzet_opr,
        "Nakłady - sprzęt ogółem": r_nak_sprzet,
        "Nakłady - sprzęt informatyczny": r_nak_sprzet_inf,
        "Nakłady - sprzęt telekomunikacyjny": r_nak_sprzet_tele,
        "Nakłady - leasing": r_nak_leasing,
        "Nakłady - oprogramowanie": r_nak_oprog,
    }
    selected_page = st.sidebar.selectbox("Wybierz analizę", pages.keys())
    pages[selected_page]()
    
def r_nak_sprzet():
#     file_path = "C:/Users/Dell/Desktop/streamlit/DATAFRAME/działy.xlsx"
#     df = pd.read_excel(file_path)
    df = pd.read_excel('działy.xlsx')

    # -----------------------------
    # MODEL 1 – WIELE LAT (2015–2021)
    # -----------------------------
    y = df["sprzet_o22"]
    X_multi = df[["sprzet_o15", "sprzet_o16", "sprzet_o17", "sprzet_o18", "sprzet_o19", "sprzet_o20", "sprzet_o21"]]
    X_multi = sm.add_constant(X_multi)
    model_multi = sm.OLS(y, X_multi).fit()

    intercept_multi = model_multi.params["const"]
    coefficients_multi = model_multi.params.drop("const")
    equation_multi = f"y = {intercept_multi:.2f}"
    for var, coef in coefficients_multi.items():
        equation_multi += f" + ({coef:.2f} * {var})"

    st.subheader("Model 1 – Regresja wieloraka (2015–2021):")
    st.text(equation_multi)
    st.code(model_multi.summary().as_text())

    # -----------------------------
    # MODEL 2 – REGRESJA PROSTA (2021)
    # -----------------------------
    X_single = df[["sprzet_o21"]]
    X_single = sm.add_constant(X_single)
    model_single = sm.OLS(y, X_single).fit()

    intercept_single = model_single.params["const"]
    coefficients_single = model_single.params.drop("const")
    equation_single = f"y = {intercept_single:.2f}"
    for var, coef in coefficients_single.items():
        equation_single += f" + ({coef:.2f} * {var})"

    st.subheader("Model 2 – Regresja liniowa (2021):")
    st.text(equation_single)
    st.code(model_single.summary().as_text())

    # -----------------------------
    # WYKRESY: Równanie jako tekst i predykcje dwóch modeli
    # -----------------------------

    st.subheader("Porównanie predykcji obu modeli:")

    # Predykcje
    y_pred_multi = model_multi.predict(X_multi)
    y_pred_single = model_single.predict(X_single)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(y, y_pred_multi, color='royalblue', alpha=0.7, label='Model wieloraki (2015–2021)')
    ax.scatter(y, y_pred_single, color='orange', alpha=0.7, label='Model prosty (2021)')
    ax.plot([y.min(), y.max()], [y.min(), y.max()], color='red', linestyle='--', label='Linia idealna')
    ax.set_xlabel("Wartości rzeczywiste (y)")
    ax.set_ylabel("Wartości przewidywane (ŷ)")
    ax.set_title("Rzeczywiste vs Przewidywane wartości – oba modele")
    ax.legend()
    ax.grid(True)
    st.pyplot(fig)
    
def r_nak_sprzet_opr():
    # file_path =  "C:/Users/Dell/Desktop/streamlit/DATAFRAME/działy.xlsx"
    # df = pd.read_excel(file_path)
    df = pd.read_excel('działy.xlsx')
    # Zmienna zależna
    y = df["sprzet_opr22"]
    
    # Zmienne niezależne: wcześniejsze lata tej samej zmiennej (2015–2021)
    X = df[["sprzet_opr15", "sprzet_opr16", "sprzet_opr17", "sprzet_opr18", "sprzet_opr19", "sprzet_opr20", "sprzet_opr21"]]
    X = sm.add_constant(X)  # dodaj stałą (intercept)
    
    # Dopasowanie modelu
    model = sm.OLS(y, X).fit()
    
    # Pobranie współczynników
    intercept = model.params["const"]
    coefficients = model.params.drop("const")
    
    # Utworzenie równania regresji
    equation = f"y = {intercept:.2f}"
    for var, coef in coefficients.items():
        equation += f" + ({coef:.2f} * {var})"
    
    # Wyświetlenie równania w Streamlit
    st.subheader("Równanie regresji liniowej:")
    st.text(equation)
    
    # Przewidywane wartości
    y_pred = model.predict(X)
    
    # Wykres punktowy
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(y, y_pred, color='royalblue', alpha=0.7)
    ax.plot([y.min(), y.max()], [y.min(), y.max()], color='red', linestyle='--')  # linia idealna
    ax.set_xlabel("Wartości rzeczywiste (y)")
    ax.set_ylabel("Wartości przewidywane (ŷ)")
    ax.set_title("Rzeczywiste vs Przewidywane wartości")
    ax.grid(True)
    st.pyplot(fig)
    # Podsumowanie modelu
    st.subheader("Podsumowanie modelu regresji:")
    st.code(model.summary().as_text())
    
def r_nak_sprzet_inf():
    # file_path =  "C:/Users/Dell/Desktop/streamlit/DATAFRAME/działy.xlsx"
    # df = pd.read_excel(file_path)
    df = pd.read_excel('działy.xlsx')
    # Zmienna zależna
    y = df["sprzet_inf22"]
    
    # Zmienne niezależne: wcześniejsze lata tej samej zmiennej (2015–2021)
    X = df[["sprzet_inf15", "sprzet_inf16", "sprzet_inf17", "sprzet_inf18", "sprzet_inf19", "sprzet_inf20", "sprzet_inf21"]]
    X = sm.add_constant(X)  # dodaj stałą (intercept)
    
    # Dopasowanie modelu
    model = sm.OLS(y, X).fit()
    
    # Pobranie współczynników
    intercept = model.params["const"]
    coefficients = model.params.drop("const")
    
    # Utworzenie równania regresji
    equation = f"y = {intercept:.2f}"
    for var, coef in coefficients.items():
        equation += f" + ({coef:.2f} * {var})"
    
    # Wyświetlenie równania w Streamlit
    st.subheader("Równanie regresji liniowej:")
    st.text(equation)
    
    # Przewidywane wartości
    y_pred = model.predict(X)
    
    # Wykres punktowy
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(y, y_pred, color='royalblue', alpha=0.7)
    ax.plot([y.min(), y.max()], [y.min(), y.max()], color='red', linestyle='--')  # linia idealna
    ax.set_xlabel("Wartości rzeczywiste (y)")
    ax.set_ylabel("Wartości przewidywane (ŷ)")
    ax.set_title("Rzeczywiste vs Przewidywane wartości")
    ax.grid(True)
    st.pyplot(fig)
    # Podsumowanie modelu
    st.subheader("Podsumowanie modelu regresji:")
    st.code(model.summary().as_text())  
    
def r_nak_sprzet_tele():
    # file_path =  "C:/Users/Dell/Desktop/streamlit/DATAFRAME/działy.xlsx"
    # df = pd.read_excel(file_path)
    df = pd.read_excel('działy.xlsx')
    
    # Zmienna zależna
    y = df["sprzet_tele22"]
    
    # Zmienne niezależne: wcześniejsze lata tej samej zmiennej (2015–2021)
    X = df[["sprzet_tele15", "sprzet_tele16", "sprzet_tele17", "sprzet_tele18", "sprzet_tele19", "sprzet_tele20", "sprzet_tele21"]]
    X = sm.add_constant(X)  # dodaj stałą (intercept)
    
    # Dopasowanie modelu
    model = sm.OLS(y, X).fit()
    
    # Pobranie współczynników
    intercept = model.params["const"]
    coefficients = model.params.drop("const")
    
    # Utworzenie równania regresji
    equation = f"y = {intercept:.2f}"
    for var, coef in coefficients.items():
        equation += f" + ({coef:.2f} * {var})"
    
    # Wyświetlenie równania w Streamlit
    st.subheader("Równanie regresji liniowej:")
    st.text(equation)
    
    # Przewidywane wartości
    y_pred = model.predict(X)
    
    # Wykres punktowy
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(y, y_pred, color='royalblue', alpha=0.7)
    ax.plot([y.min(), y.max()], [y.min(), y.max()], color='red', linestyle='--')  # linia idealna
    ax.set_xlabel("Wartości rzeczywiste (y)")
    ax.set_ylabel("Wartości przewidywane (ŷ)")
    ax.set_title("Rzeczywiste vs Przewidywane wartości")
    ax.grid(True)
    st.pyplot(fig)
    # Podsumowanie modelu
    st.subheader("Podsumowanie modelu regresji:")
    st.code(model.summary().as_text())   
    
def r_nak_leasing():
    # file_path =  "C:/Users/Dell/Desktop/streamlit/DATAFRAME/działy.xlsx"
    # df = pd.read_excel(file_path)
    df = pd.read_excel('działy.xlsx')
    
    # Zmienna zależna
    y = df["leasing22"]
    
    # Zmienne niezależne: wcześniejsze lata tej samej zmiennej (2015–2021)
    X = df[["leasing15", "leasing16", "leasing17", "leasing18", "leasing19", "leasing20", "leasing21"]]
    X = sm.add_constant(X)  # dodaj stałą (intercept)
    
    # Dopasowanie modelu
    model = sm.OLS(y, X).fit()
    
    # Pobranie współczynników
    intercept = model.params["const"]
    coefficients = model.params.drop("const")
    
    # Utworzenie równania regresji
    equation = f"y = {intercept:.2f}"
    for var, coef in coefficients.items():
        equation += f" + ({coef:.2f} * {var})"
    
    # Wyświetlenie równania w Streamlit
    st.subheader("Równanie regresji liniowej:")
    st.text(equation)
    
    # Przewidywane wartości
    y_pred = model.predict(X)
    
    # Wykres punktowy
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(y, y_pred, color='royalblue', alpha=0.7)
    ax.plot([y.min(), y.max()], [y.min(), y.max()], color='red', linestyle='--')  # linia idealna
    ax.set_xlabel("Wartości rzeczywiste (y)")
    ax.set_ylabel("Wartości przewidywane (ŷ)")
    ax.set_title("Rzeczywiste vs Przewidywane wartości")
    ax.grid(True)
    st.pyplot(fig)
    # Podsumowanie modelu
    st.subheader("Podsumowanie modelu regresji:")
    st.code(model.summary().as_text())    
    
def r_nak_oprog():
    # file_path =  "C:/Users/Dell/Desktop/streamlit/DATAFRAME/działy.xlsx"
    # df = pd.read_excel(file_path)
    df = pd.read_excel('działy.xlsx')
    
    # Zmienna zależna
    y = df["oprog22"]
    
    # Zmienne niezależne: wcześniejsze lata tej samej zmiennej (2015–2021)
    X = df[["oprog15", "oprog16", "oprog17", "oprog18", "oprog19", "oprog20", "oprog21"]]
    X = sm.add_constant(X)  # dodaj stałą (intercept)
    
    # Dopasowanie modelu
    model = sm.OLS(y, X).fit()
    
    # Pobranie współczynników
    intercept = model.params["const"]
    coefficients = model.params.drop("const")
    
    # Utworzenie równania regresji
    equation = f"y = {intercept:.2f}"
    for var, coef in coefficients.items():
        equation += f" + ({coef:.2f} * {var})"
    
    # Wyświetlenie równania w Streamlit
    st.subheader("Równanie regresji liniowej:")
    st.text(equation)
    
    # Przewidywane wartości
    y_pred = model.predict(X)
    
    # Wykres punktowy
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(y, y_pred, color='royalblue', alpha=0.7)
    ax.plot([y.min(), y.max()], [y.min(), y.max()], color='red', linestyle='--')  # linia idealna
    ax.set_xlabel("Wartości rzeczywiste (y)")
    ax.set_ylabel("Wartości przewidywane (ŷ)")
    ax.set_title("Rzeczywiste vs Przewidywane wartości")
    ax.grid(True)
    st.pyplot(fig)
    # Podsumowanie modelu
    st.subheader("Podsumowanie modelu regresji:")
    st.code(model.summary().as_text())   
    
def r_pracownicy_section():
    st.sidebar.header("Pracownicy")
    pages = {
        "Przedsiębiorstwa zatrudniające pracowników": r_przed_o,
        "Przedsiębiorstwa z dostępem do internetu": r_przed_di,
        "Pracujący ogółem": r_pr_o,
        "Pracownicy z wyższym wykształceniem": r_pr_ww,
        "Pracownicy z dostępem do internetu": r_pr_di,
        "Pracownicy z urządzeniami przenośnymi": r_pr_up,
    }
    selected_page = st.sidebar.selectbox("Wybierz analizę", pages.keys())
    pages[selected_page]()    
    
def r_przed_o():
    # file_path =  "C:/Users/Dell/Desktop/streamlit/DATAFRAME/działy.xlsx"
    # df = pd.read_excel(file_path)
    df = pd.read_excel('działy.xlsx')
    
    # Zmienna zależna
    y = df["przed_o21"]
    
    # Zmienne niezależne: wcześniejsze lata tej samej zmiennej (2015–2021)
    X = df[["przed_o15", "przed_o16", "przed_o17", "przed_o18", "przed_o19", "przed_o20"]]
    X = sm.add_constant(X)  # dodaj stałą (intercept)
    
    # Dopasowanie modelu
    model = sm.OLS(y, X).fit()
    
    # Pobranie współczynników
    intercept = model.params["const"]
    coefficients = model.params.drop("const")
    
    # Utworzenie równania regresji
    equation = f"y = {intercept:.2f}"
    for var, coef in coefficients.items():
        equation += f" + ({coef:.2f} * {var})"
    
    # Wyświetlenie równania w Streamlit
    st.subheader("Równanie regresji liniowej:")
    st.text(equation)
    
    # Przewidywane wartości
    y_pred = model.predict(X)
    
    # Wykres punktowy
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(y, y_pred, color='royalblue', alpha=0.7)
    ax.plot([y.min(), y.max()], [y.min(), y.max()], color='red', linestyle='--')  # linia idealna
    ax.set_xlabel("Wartości rzeczywiste (y)")
    ax.set_ylabel("Wartości przewidywane (ŷ)")
    ax.set_title("Rzeczywiste vs Przewidywane wartości")
    ax.grid(True)
    st.pyplot(fig)    
    # Podsumowanie modelu
    st.subheader("Podsumowanie modelu regresji:")
    st.code(model.summary().as_text()) 

def r_przed_di():
    # file_path =  "C:/Users/Dell/Desktop/streamlit/DATAFRAME/działy.xlsx"
    # df = pd.read_excel(file_path)
    df = pd.read_excel('działy.xlsx')
    
    # Zmienna zależna
    y = df["przed_di21"]
    
    # Zmienne niezależne: wcześniejsze lata tej samej zmiennej (2015–2021)
    X = df[["przed_di15", "przed_di16", "przed_di17", "przed_di18", "przed_di19", "przed_di20"]]
    X = sm.add_constant(X)  # dodaj stałą (intercept)
    
    # Dopasowanie modelu
    model = sm.OLS(y, X).fit()
    
    # Pobranie współczynników
    intercept = model.params["const"]
    coefficients = model.params.drop("const")
    
    # Utworzenie równania regresji
    equation = f"y = {intercept:.2f}"
    for var, coef in coefficients.items():
        equation += f" + ({coef:.2f} * {var})"
    
    # Wyświetlenie równania w Streamlit
    st.subheader("Równanie regresji liniowej:")
    st.text(equation)
    
    # Przewidywane wartości
    y_pred = model.predict(X)
    
    # Wykres punktowy
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(y, y_pred, color='royalblue', alpha=0.7)
    ax.plot([y.min(), y.max()], [y.min(), y.max()], color='red', linestyle='--')  # linia idealna
    ax.set_xlabel("Wartości rzeczywiste (y)")
    ax.set_ylabel("Wartości przewidywane (ŷ)")
    ax.set_title("Rzeczywiste vs Przewidywane wartości")
    ax.grid(True)
    st.pyplot(fig)     
    # Podsumowanie modelu
    st.subheader("Podsumowanie modelu regresji:")
    st.code(model.summary().as_text())
    
def r_pr_o():
    # file_path =  "C:/Users/Dell/Desktop/streamlit/DATAFRAME/działy.xlsx"
    # df = pd.read_excel(file_path)
    df = pd.read_excel('działy.xlsx')
    
    # Zmienna zależna
    y = df["pr_o21"]
    
    # Zmienne niezależne: wcześniejsze lata tej samej zmiennej (2015–2021)
    X = df[["pr_o15", "pr_o16", "pr_o17", "pr_o18", "pr_o19", "pr_o20"]]
    X = sm.add_constant(X)  # dodaj stałą (intercept)
    
    # Dopasowanie modelu
    model = sm.OLS(y, X).fit()
    
    # Pobranie współczynników
    intercept = model.params["const"]
    coefficients = model.params.drop("const")
    
    # Utworzenie równania regresji
    equation = f"y = {intercept:.2f}"
    for var, coef in coefficients.items():
        equation += f" + ({coef:.2f} * {var})"
    
    # Wyświetlenie równania w Streamlit
    st.subheader("Równanie regresji liniowej:")
    st.text(equation)
    
    # Przewidywane wartości
    y_pred = model.predict(X)
    
    # Wykres punktowy
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(y, y_pred, color='royalblue', alpha=0.7)
    ax.plot([y.min(), y.max()], [y.min(), y.max()], color='red', linestyle='--')  # linia idealna
    ax.set_xlabel("Wartości rzeczywiste (y)")
    ax.set_ylabel("Wartości przewidywane (ŷ)")
    ax.set_title("Rzeczywiste vs Przewidywane wartości")
    ax.grid(True)
    st.pyplot(fig)   
    # Podsumowanie modelu
    st.subheader("Podsumowanie modelu regresji:")
    st.code(model.summary().as_text())
    
def r_pr_ww():
    # file_path =  "C:/Users/Dell/Desktop/streamlit/DATAFRAME/działy.xlsx"
    # df = pd.read_excel(file_path)
    df = pd.read_excel('działy.xlsx')
    
    # Zmienna zależna
    y = df["pr_ww21"]
    
    # Zmienne niezależne: wcześniejsze lata tej samej zmiennej (2015–2021)
    X = df[["pr_ww15", "pr_ww16", "pr_ww17", "pr_ww18", "pr_ww19", "pr_ww20"]]
    X = sm.add_constant(X)  # dodaj stałą (intercept)
    
    # Dopasowanie modelu
    model = sm.OLS(y, X).fit()
    
    # Pobranie współczynników
    intercept = model.params["const"]
    coefficients = model.params.drop("const")
    
    # Utworzenie równania regresji
    equation = f"y = {intercept:.2f}"
    for var, coef in coefficients.items():
        equation += f" + ({coef:.2f} * {var})"
    
    # Wyświetlenie równania w Streamlit
    st.subheader("Równanie regresji liniowej:")
    st.text(equation)
    
    # Przewidywane wartości
    y_pred = model.predict(X)
    
    # Wykres punktowy
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(y, y_pred, color='royalblue', alpha=0.7)
    ax.plot([y.min(), y.max()], [y.min(), y.max()], color='red', linestyle='--')  # linia idealna
    ax.set_xlabel("Wartości rzeczywiste (y)")
    ax.set_ylabel("Wartości przewidywane (ŷ)")
    ax.set_title("Rzeczywiste vs Przewidywane wartości")
    ax.grid(True)
    st.pyplot(fig)   
    # Podsumowanie modelu
    st.subheader("Podsumowanie modelu regresji:")
    st.code(model.summary().as_text())
    
def r_pr_up():
    # file_path =  "C:/Users/Dell/Desktop/streamlit/DATAFRAME/działy.xlsx"
    # df = pd.read_excel(file_path)
    df = pd.read_excel('działy.xlsx')
    
    # Zmienna zależna
    y = df["pr_up21"]
    
    # Zmienne niezależne: wcześniejsze lata tej samej zmiennej (2015–2021)
    X = df[["pr_up15", "pr_up16", "pr_up17", "pr_up18", "pr_up19", "pr_up20"]]
    X = sm.add_constant(X)  # dodaj stałą (intercept)
    
    # Dopasowanie modelu
    model = sm.OLS(y, X).fit()
    
    # Pobranie współczynników
    intercept = model.params["const"]
    coefficients = model.params.drop("const")
    
    # Utworzenie równania regresji
    equation = f"y = {intercept:.2f}"
    for var, coef in coefficients.items():
        equation += f" + ({coef:.2f} * {var})"
    
    # Wyświetlenie równania w Streamlit
    st.subheader("Równanie regresji liniowej:")
    st.text(equation)
    
    # Przewidywane wartości
    y_pred = model.predict(X)
    
    # Wykres punktowy
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(y, y_pred, color='royalblue', alpha=0.7)
    ax.plot([y.min(), y.max()], [y.min(), y.max()], color='red', linestyle='--')  # linia idealna
    ax.set_xlabel("Wartości rzeczywiste (y)")
    ax.set_ylabel("Wartości przewidywane (ŷ)")
    ax.set_title("Rzeczywiste vs Przewidywane wartości")
    ax.grid(True)
    st.pyplot(fig)   
    # Podsumowanie modelu
    st.subheader("Podsumowanie modelu regresji:")
    st.code(model.summary().as_text())
    
def r_pr_di():
    # file_path =  "C:/Users/Dell/Desktop/streamlit/DATAFRAME/działy.xlsx"
    # df = pd.read_excel(file_path)
    df = pd.read_excel('działy.xlsx')
    
    # Zmienna zależna
    y = df["pr_di21"]
    
    # Zmienne niezależne: wcześniejsze lata tej samej zmiennej (2015–2021)
    X = df[["pr_di15", "pr_di16", "pr_di17", "pr_di18", "pr_di19", "pr_di20"]]
    X = sm.add_constant(X)  # dodaj stałą (intercept)
    
    # Dopasowanie modelu
    model = sm.OLS(y, X).fit()
    
    # Pobranie współczynników
    intercept = model.params["const"]
    coefficients = model.params.drop("const")
    
    # Utworzenie równania regresji
    equation = f"y = {intercept:.2f}"
    for var, coef in coefficients.items():
        equation += f" + ({coef:.2f} * {var})"
    
    # Wyświetlenie równania w Streamlit
    st.subheader("Równanie regresji liniowej:")
    st.text(equation)
    
    # Przewidywane wartości
    y_pred = model.predict(X)
    
    # Wykres punktowy
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(y, y_pred, color='royalblue', alpha=0.7)
    ax.plot([y.min(), y.max()], [y.min(), y.max()], color='red', linestyle='--')  # linia idealna
    ax.set_xlabel("Wartości rzeczywiste (y)")
    ax.set_ylabel("Wartości przewidywane (ŷ)")
    ax.set_title("Rzeczywiste vs Przewidywane wartości")
    ax.grid(True)
    st.pyplot(fig)  
    # Podsumowanie modelu
    st.subheader("Podsumowanie modelu regresji:")
    st.code(model.summary().as_text())
    
def r_wartosc_nakladow_section():
    st.sidebar.header("Wartość Nakładów")
    pages = {
        "Wartość nakładów - sprzęt/oprogramowanie": r_wnak_sprzet_opr,
        "Wartość nakładów - sprzęt": r_wnak_sprzet,
        "Wartość nakładów - sprzęt informatyczny": r_wnak_sprzet_inf,
        "Wartość nakładów - sprzęt teleinformatyczny": r_wnak_sprzet_tele,
        "Wartość nakładów - leasing": r_wnak_leasing,
        "Wartość nakładów - oprogramowanie": r_wnak_oprog,
    }
    selected_page = st.sidebar.selectbox("Wybierz analizę", pages.keys())
    pages[selected_page]()
    
    
def r_wnak_sprzet():
    # file_path =  "C:/Users/Dell/Desktop/streamlit/DATAFRAME/działy.xlsx"
    # df = pd.read_excel(file_path)
    df = pd.read_excel('działy.xlsx')
    
    # Zmienna zależna
    y = df["wsprzet_o22"]
    
    # Zmienne niezależne: wcześniejsze lata tej samej zmiennej (2015–2021)
    X = df[["wsprzet_o15", "wsprzet_o16", "wsprzet_o17", "wsprzet_o18", "wsprzet_o19", "wsprzet_o20", "wsprzet_o21"]]
    X = sm.add_constant(X)  # dodaj stałą (intercept)
    
    # Dopasowanie modelu
    model = sm.OLS(y, X).fit()
    
    # Pobranie współczynników
    intercept = model.params["const"]
    coefficients = model.params.drop("const")
    
    # Utworzenie równania regresji
    equation = f"y = {intercept:.2f}"
    for var, coef in coefficients.items():
        equation += f" + ({coef:.2f} * {var})"
    
    # Wyświetlenie równania w Streamlit
    st.subheader("Równanie regresji liniowej:")
    st.text(equation)
    
    # Przewidywane wartości
    y_pred = model.predict(X)
    
    # Wykres punktowy
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(y, y_pred, color='royalblue', alpha=0.7)
    ax.plot([y.min(), y.max()], [y.min(), y.max()], color='red', linestyle='--')  # linia idealna
    ax.set_xlabel("Wartości rzeczywiste (y)")
    ax.set_ylabel("Wartości przewidywane (ŷ)")
    ax.set_title("Rzeczywiste vs Przewidywane wartości")
    ax.grid(True)
    st.pyplot(fig)
    # Podsumowanie modelu
    st.subheader("Podsumowanie modelu regresji:")
    st.code(model.summary().as_text())
    
def r_wnak_sprzet_opr():
    # file_path =  "C:/Users/Dell/Desktop/streamlit/DATAFRAME/działy.xlsx"
    # df = pd.read_excel(file_path)
    df = pd.read_excel('działy.xlsx')
    
    # Zmienna zależna
    y = df["wsprzet_opr22"]
    
    # Zmienne niezależne: wcześniejsze lata tej samej zmiennej (2015–2021)
    X = df[["wsprzet_opr15", "wsprzet_opr16", "wsprzet_opr17", "wsprzet_opr18", "wsprzet_opr19", "wsprzet_opr20", "wsprzet_opr21"]]
    X = sm.add_constant(X)  # dodaj stałą (intercept)
    
    # Dopasowanie modelu
    model = sm.OLS(y, X).fit()
    
    # Pobranie współczynników
    intercept = model.params["const"]
    coefficients = model.params.drop("const")
    
    # Utworzenie równania regresji
    equation = f"y = {intercept:.2f}"
    for var, coef in coefficients.items():
        equation += f" + ({coef:.2f} * {var})"
    
    # Wyświetlenie równania w Streamlit
    st.subheader("Równanie regresji liniowej:")
    st.text(equation)
    
    # Przewidywane wartości
    y_pred = model.predict(X)
    
    # Wykres punktowy
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(y, y_pred, color='royalblue', alpha=0.7)
    ax.plot([y.min(), y.max()], [y.min(), y.max()], color='red', linestyle='--')  # linia idealna
    ax.set_xlabel("Wartości rzeczywiste (y)")
    ax.set_ylabel("Wartości przewidywane (ŷ)")
    ax.set_title("Rzeczywiste vs Przewidywane wartości")
    ax.grid(True)
    st.pyplot(fig)
    # Podsumowanie modelu
    st.subheader("Podsumowanie modelu regresji:")
    st.code(model.summary().as_text())
    
def r_wnak_sprzet_inf():
    # file_path =  "C:/Users/Dell/Desktop/streamlit/DATAFRAME/działy.xlsx"
    # df = pd.read_excel(file_path)
    df = pd.read_excel('działy.xlsx')
    
    # Zmienna zależna
    y = df["wsprzet_inf22"]
    
    # Zmienne niezależne: wcześniejsze lata tej samej zmiennej (2015–2021)
    X = df[["wsprzet_inf15", "wsprzet_inf16", "wsprzet_inf17", "wsprzet_inf18", "wsprzet_inf19", "wsprzet_inf20", "wsprzet_inf21"]]
    X = sm.add_constant(X)  # dodaj stałą (intercept)
    
    # Dopasowanie modelu
    model = sm.OLS(y, X).fit()
    
    # Pobranie współczynników
    intercept = model.params["const"]
    coefficients = model.params.drop("const")
    
    # Utworzenie równania regresji
    equation = f"y = {intercept:.2f}"
    for var, coef in coefficients.items():
        equation += f" + ({coef:.2f} * {var})"
    
    # Wyświetlenie równania w Streamlit
    st.subheader("Równanie regresji liniowej:")
    st.text(equation)
    
    # Przewidywane wartości
    y_pred = model.predict(X)
    
    # Wykres punktowy
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(y, y_pred, color='royalblue', alpha=0.7)
    ax.plot([y.min(), y.max()], [y.min(), y.max()], color='red', linestyle='--')  # linia idealna
    ax.set_xlabel("Wartości rzeczywiste (y)")
    ax.set_ylabel("Wartości przewidywane (ŷ)")
    ax.set_title("Rzeczywiste vs Przewidywane wartości")
    ax.grid(True)
    st.pyplot(fig)
    # Podsumowanie modelu
    st.subheader("Podsumowanie modelu regresji:")
    st.code(model.summary().as_text())
    
def r_wnak_sprzet_tele():
    # file_path =  "C:/Users/Dell/Desktop/streamlit/DATAFRAME/działy.xlsx"
    # df = pd.read_excel(file_path)
    df = pd.read_excel('działy.xlsx')
    
    # Zmienna zależna
    y = df["wsprzet_tele22"]
    
    # Zmienne niezależne: wcześniejsze lata tej samej zmiennej (2015–2021)
    X = df[["wsprzet_tele15", "wsprzet_tele16", "wsprzet_tele17", "wsprzet_tele18", "wsprzet_tele19", "wsprzet_tele20", "wsprzet_tele21"]]
    X = sm.add_constant(X)  # dodaj stałą (intercept)
    
    # Dopasowanie modelu
    model = sm.OLS(y, X).fit()
    
    # Pobranie współczynników
    intercept = model.params["const"]
    coefficients = model.params.drop("const")
    
    # Utworzenie równania regresji
    equation = f"y = {intercept:.2f}"
    for var, coef in coefficients.items():
        equation += f" + ({coef:.2f} * {var})"
    
    # Wyświetlenie równania w Streamlit
    st.subheader("Równanie regresji liniowej:")
    st.text(equation)
    
    # Przewidywane wartości
    y_pred = model.predict(X)
    
    # Wykres punktowy
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(y, y_pred, color='royalblue', alpha=0.7)
    ax.plot([y.min(), y.max()], [y.min(), y.max()], color='red', linestyle='--')  # linia idealna
    ax.set_xlabel("Wartości rzeczywiste (y)")
    ax.set_ylabel("Wartości przewidywane (ŷ)")
    ax.set_title("Rzeczywiste vs Przewidywane wartości")
    ax.grid(True)
    st.pyplot(fig)
    # Podsumowanie modelu
    st.subheader("Podsumowanie modelu regresji:")
    st.code(model.summary().as_text())
    
def r_wnak_leasing():
    # file_path =  "C:/Users/Dell/Desktop/streamlit/DATAFRAME/działy.xlsx"
    # df = pd.read_excel(file_path)
    df = pd.read_excel('działy.xlsx')
    
    # Zmienna zależna
    y = df["wleasing22"]
    
    # Zmienne niezależne: wcześniejsze lata tej samej zmiennej (2015–2021)
    X = df[["wleasing15", "wleasing16", "wleasing17", "wleasing18", "wleasing19", "wleasing20", "wleasing21"]]
    X = sm.add_constant(X)  # dodaj stałą (intercept)
    
    # Dopasowanie modelu
    model = sm.OLS(y, X).fit()
    
    # Pobranie współczynników
    intercept = model.params["const"]
    coefficients = model.params.drop("const")
    
    # Utworzenie równania regresji
    equation = f"y = {intercept:.2f}"
    for var, coef in coefficients.items():
        equation += f" + ({coef:.2f} * {var})"
    
    # Wyświetlenie równania w Streamlit
    st.subheader("Równanie regresji liniowej:")
    st.text(equation)
    
    # Przewidywane wartości
    y_pred = model.predict(X)
    
    # Wykres punktowy
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(y, y_pred, color='royalblue', alpha=0.7)
    ax.plot([y.min(), y.max()], [y.min(), y.max()], color='red', linestyle='--')  # linia idealna
    ax.set_xlabel("Wartości rzeczywiste (y)")
    ax.set_ylabel("Wartości przewidywane (ŷ)")
    ax.set_title("Rzeczywiste vs Przewidywane wartości")
    ax.grid(True)
    st.pyplot(fig)
    # Podsumowanie modelu
    st.subheader("Podsumowanie modelu regresji:")
    st.code(model.summary().as_text())
    
def r_wnak_oprog():
    # file_path =  "C:/Users/Dell/Desktop/streamlit/DATAFRAME/działy.xlsx"
    # df = pd.read_excel(file_path)
    df = pd.read_excel('działy.xlsx')
    
    # Zmienna zależna
    y = df["woprog22"]
    
    # Zmienne niezależne: wcześniejsze lata tej samej zmiennej (2015–2021)
    X = df[["woprog15", "woprog16", "woprog17", "woprog18", "woprog19", "woprog20", "woprog21"]]
    X = sm.add_constant(X)  # dodaj stałą (intercept)
    
    # Dopasowanie modelu
    model = sm.OLS(y, X).fit()
    
    # Pobranie współczynników
    intercept = model.params["const"]
    coefficients = model.params.drop("const")
    
    # Utworzenie równania regresji
    equation = f"y = {intercept:.2f}"
    for var, coef in coefficients.items():
        equation += f" + ({coef:.2f} * {var})"
    
    # Wyświetlenie równania w Streamlit
    st.subheader("Równanie regresji liniowej:")
    st.text(equation)
    
    # Przewidywane wartości
    y_pred = model.predict(X)
    
    # Wykres punktowy
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(y, y_pred, color='royalblue', alpha=0.7)
    ax.plot([y.min(), y.max()], [y.min(), y.max()], color='red', linestyle='--')  # linia idealna
    ax.set_xlabel("Wartości rzeczywiste (y)")
    ax.set_ylabel("Wartości przewidywane (ŷ)")
    ax.set_title("Rzeczywiste vs Przewidywane wartości")
    ax.grid(True)
    st.pyplot(fig)
    # Podsumowanie modelu
    st.subheader("Podsumowanie modelu regresji:")
    st.code(model.summary().as_text())
    
    
def r_rodzaje_polaczen_section():
    st.sidebar.header("Rodzaje Połączeń")
    pages = {
        #"Przedsiębiorstwa prowadzące działalność ogółem": maps_demo_rodz_polaczen,
        "szerokopasmowy  dostęp do Internetu": r_rodz_pol_szer,
        "szerokopasmowy  dostęp do Internetu poprzez łącze DSL": r_rodz_pol_szerDSL,
    }
    selected_page = st.sidebar.selectbox("Wybierz analizę", pages.keys())
    pages[selected_page]()  
    
def r_rodz_pol_szer():
    # file_path =  "C:/Users/Dell/Desktop/streamlit/DATAFRAME/działy.xlsx"
    # df = pd.read_excel(file_path)
    df = pd.read_excel('działy.xlsx')
    
    # Zmienna zależna
    y = df["od_szer_o22"]
    
    # Zmienne niezależne: wcześniejsze lata tej samej zmiennej (2015–2021)
    X = df[["od_szer_o15", "od_szer_o16", "od_szer_o17", "od_szer_o18", "od_szer_o19", "od_szer_o20", "od_szer_o21"]]
    X = sm.add_constant(X)  # dodaj stałą (intercept)
    
    # Dopasowanie modelu
    model = sm.OLS(y, X).fit()
    
    # Pobranie współczynników
    intercept = model.params["const"]
    coefficients = model.params.drop("const")
    
    # Utworzenie równania regresji
    equation = f"y = {intercept:.2f}"
    for var, coef in coefficients.items():
        equation += f" + ({coef:.2f} * {var})"
    
    # Wyświetlenie równania w Streamlit
    st.subheader("Równanie regresji liniowej:")
    st.text(equation)
    
    # Przewidywane wartości
    y_pred = model.predict(X)
    
    # Wykres punktowy
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(y, y_pred, color='royalblue', alpha=0.7)
    ax.plot([y.min(), y.max()], [y.min(), y.max()], color='red', linestyle='--')  # linia idealna
    ax.set_xlabel("Wartości rzeczywiste (y)")
    ax.set_ylabel("Wartości przewidywane (ŷ)")
    ax.set_title("Rzeczywiste vs Przewidywane wartości")
    ax.grid(True)
    st.pyplot(fig)
    # Podsumowanie modelu
    st.subheader("Podsumowanie modelu regresji:")
    st.code(model.summary().as_text())
    
def r_rodz_pol_szerDSL():
    # file_path =  "C:/Users/Dell/Desktop/streamlit/DATAFRAME/działy.xlsx"
    # df = pd.read_excel(file_path)
    df = pd.read_excel('działy.xlsx')
    
    # Zmienna zależna
    y = df["od_szer_DSL22"]
    
    # Zmienne niezależne: wcześniejsze lata tej samej zmiennej (2015–2021)
    X = df[["od_szer_DSL15", "od_szer_DSL16", "od_szer_DSL17", "od_szer_DSL18", "od_szer_DSL19", "od_szer_DSL20", "od_szer_DSL21"]]
    X = sm.add_constant(X)  # dodaj stałą (intercept)
    
    # Dopasowanie modelu
    model = sm.OLS(y, X).fit()
    
    # Pobranie współczynników
    intercept = model.params["const"]
    coefficients = model.params.drop("const")
    
    # Utworzenie równania regresji
    equation = f"y = {intercept:.2f}"
    for var, coef in coefficients.items():
        equation += f" + ({coef:.2f} * {var})"
    
    # Wyświetlenie równania w Streamlit
    st.subheader("Równanie regresji liniowej:")
    st.text(equation)
    
    # Przewidywane wartości
    y_pred = model.predict(X)
    
    # Wykres punktowy
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(y, y_pred, color='royalblue', alpha=0.7)
    ax.plot([y.min(), y.max()], [y.min(), y.max()], color='red', linestyle='--')  # linia idealna
    ax.set_xlabel("Wartości rzeczywiste (y)")
    ax.set_ylabel("Wartości przewidywane (ŷ)")
    ax.set_title("Rzeczywiste vs Przewidywane wartości")
    ax.grid(True)
    st.pyplot(fig)
    # Podsumowanie modelu
    st.subheader("Podsumowanie modelu regresji:")
    st.code(model.summary().as_text())
    
def r_predkosc_polaczen_section():
    st.sidebar.header("Prędkość Połączeń")
    pages = {
        "szerokopasmowy  dostęp do Internetu - prędkość połączenia mniej niż 100 Mbit/s": r_pred_pol_szer_w,
        "szerokopasmowy  dostęp do Internetu - prędkość połączenia przynajmniej 100 Mbit/s": r_pred_pol_szer_s,
    }
    selected_page = st.sidebar.selectbox("Wybierz analizę", pages.keys())
    pages[selected_page]()
    
def r_pred_pol_szer_w():
    # file_path =  "C:/Users/Dell/Desktop/streamlit/DATAFRAME/działy.xlsx"
    # df = pd.read_excel(file_path)
    df = pd.read_excel('działy.xlsx')
    
    # Zmienna zależna
    y = df["od_szer_w22"]
    
    # Zmienne niezależne: wcześniejsze lata tej samej zmiennej (2015–2021)
    X = df[["od_szer_w15", "od_szer_w16", "od_szer_w17", "od_szer_w18", "od_szer_w19", "od_szer_w20", "od_szer_w21"]]
    X = sm.add_constant(X)  # dodaj stałą (intercept)
    
    # Dopasowanie modelu
    model = sm.OLS(y, X).fit()
    
    # Pobranie współczynników
    intercept = model.params["const"]
    coefficients = model.params.drop("const")
    
    # Utworzenie równania regresji
    equation = f"y = {intercept:.2f}"
    for var, coef in coefficients.items():
        equation += f" + ({coef:.2f} * {var})"
    
    # Wyświetlenie równania w Streamlit
    st.subheader("Równanie regresji liniowej:")
    st.text(equation)
    
    # Przewidywane wartości
    y_pred = model.predict(X)
    
    # Wykres punktowy
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(y, y_pred, color='royalblue', alpha=0.7)
    ax.plot([y.min(), y.max()], [y.min(), y.max()], color='red', linestyle='--')  # linia idealna
    ax.set_xlabel("Wartości rzeczywiste (y)")
    ax.set_ylabel("Wartości przewidywane (ŷ)")
    ax.set_title("Rzeczywiste vs Przewidywane wartości")
    ax.grid(True)
    st.pyplot(fig)
    # Podsumowanie modelu
    st.subheader("Podsumowanie modelu regresji:")
    st.code(model.summary().as_text())
    
def r_pred_pol_szer_s():
    # file_path =  "C:/Users/Dell/Desktop/streamlit/DATAFRAME/działy.xlsx"
    # df = pd.read_excel(file_path)
    df = pd.read_excel('działy.xlsx')
    
    # Zmienna zależna
    y = df["od_szer_s22"]
    
    # Zmienne niezależne: wcześniejsze lata tej samej zmiennej (2015–2021)
    X = df[["od_szer_s15", "od_szer_s16", "od_szer_s17", "od_szer_s18", "od_szer_s19", "od_szer_s20", "od_szer_s21"]]
    X = sm.add_constant(X)  # dodaj stałą (intercept)
    
    # Dopasowanie modelu
    model = sm.OLS(y, X).fit()
    
    # Pobranie współczynników
    intercept = model.params["const"]
    coefficients = model.params.drop("const")
    
    # Utworzenie równania regresji
    equation = f"y = {intercept:.2f}"
    for var, coef in coefficients.items():
        equation += f" + ({coef:.2f} * {var})"
    
    # Wyświetlenie równania w Streamlit
    st.subheader("Równanie regresji liniowej:")
    st.text(equation)
    
    # Przewidywane wartości
    y_pred = model.predict(X)
    
    # Wykres punktowy
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(y, y_pred, color='royalblue', alpha=0.7)
    ax.plot([y.min(), y.max()], [y.min(), y.max()], color='red', linestyle='--')  # linia idealna
    ax.set_xlabel("Wartości rzeczywiste (y)")
    ax.set_ylabel("Wartości przewidywane (ŷ)")
    ax.set_title("Rzeczywiste vs Przewidywane wartości")
    ax.grid(True)
    st.pyplot(fig)
    # Podsumowanie modelu
    st.subheader("Podsumowanie modelu regresji:")
    st.code(model.summary().as_text())
    
############################KORELACJE - HANDEL#######################################################################################
def korelacje_handel_section():
    selected_subsection = st.sidebar.selectbox("Wybierz kategorię", [
        "Nakłady", "Pracownicy", "Wartość Nakładów", "Rodzaje Połączeń", "Prędkość Połączeń"
    ])
    if selected_subsection == "Nakłady":
        handel_corr_naklady_section()
    elif selected_subsection == "Pracownicy":
        handel_corr_pracownicy_section()
    elif selected_subsection == "Wartość Nakładów":
        handel_corr_wartosc_nakladow_section()
    elif selected_subsection == "Rodzaje Połączeń":
        handel_corr_rodzaje_polaczen_section()
    elif selected_subsection == "Prędkość Połączeń":
        handel_corr_predkosc_polaczen_section()
        
def handel_corr_naklady_section():
    st.sidebar.header("Nakłady")
    pages = {
        "Nakłady na sprzęt/oprogramowanie": handel_nak_sprzet_opr_corr,
        "Nakłady - sprzęt ogółem": handel_nak_sprzet_corr,
        "Nakłady - sprzęt informatyczny": handel_nak_sprzet_inf_corr,
        "Nakłady - sprzęt telekomunikacyjny": handel_nak_sprzet_tele_corr,
        "Nakłady - leasing": handel_nak_leasing_corr,
        "Nakłady - oprogramowanie": handel_nak_oprog_corr,
    }
    selected_page = st.sidebar.selectbox("Wybierz analizę", pages.keys())
    pages[selected_page]()
    
def handel_nak_leasing_corr():
    st.markdown('### Korelacje - przedsiębiorstwa ponoszące nakłady na leasing urządzeń ICT w branżach związanych z handlem')
    
    # plik_wejsciowy =  "C:/Users/Dell/Desktop/streamlit/DATAFRAME/datadzialy_wide_N_leasing.xlsx" 
    # df = pd.read_excel(plik_wejsciowy)
    df = pd.read_excel('datadzialy_wide_N_leasing.xlsx')
    
    handel=df[df["wyszczegolnienie"].isin([
    'Handel i naprawy',
    'Handel i naprawa pojazdów',
    'Handel hurtowy',
    'Sprzedaż hurtowa narzędzi technologii informacyjnej i komunikacyjnej',
    'Handel detaliczny'])]
    vector_handel = handel["wyszczegolnienie"].tolist()
    df_transformed_handel = handel.set_index("wyszczegolnienie").T
    df_transformed_handel.columns = vector_handel
    df_transformed_handel.index = [2015, 2016, 2017, 2018, 2019, 2020, 2021,2022]
    correlation_matrix = df_transformed_handel.corr()
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
    plt.title("Macierz korelacji")
    plt.show()
    st.pyplot(fig)
    
def handel_nak_sprzet_opr_corr():
    st.markdown('### Korelacje - przedsiębiorstwa ponoszące nakłady na sprzęt i oprogramowanie w branżach związanych z handlem')
    
    # plik_wejsciowy =  "C:/Users/Dell/Desktop/streamlit/DATAFRAME/datadzialy_wide_N_sprzet_opr.xlsx" 
    # df = pd.read_excel(plik_wejsciowy)
    df = pd.read_excel('datadzialy_wide_N_sprzet_opr.xlsx')
    
    handel=df[df["wyszczegolnienie"].isin([
    'Handel i naprawy',
    'Handel i naprawa pojazdów',
    'Handel hurtowy',
    'Sprzedaż hurtowa narzędzi technologii informacyjnej i komunikacyjnej',
    'Handel detaliczny'])]
    vector_handel = handel["wyszczegolnienie"].tolist()
    df_transformed_handel = handel.set_index("wyszczegolnienie").T
    df_transformed_handel.columns = vector_handel
    df_transformed_handel.index = [2015, 2016, 2017, 2018, 2019, 2020, 2021,2022]
    correlation_matrix = df_transformed_handel.corr()
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
    plt.title("Macierz korelacji")
    plt.show()
    st.pyplot(fig)
    
def handel_nak_sprzet_corr():
    st.markdown('### Korelacje - przedsiębiorstwa ponoszące nakłady na sprzęt w branżach związanych z handlem')
    
    # plik_wejsciowy =  "C:/Users/Dell/Desktop/streamlit/DATAFRAME/datadzialy_wide_N_sprzet_o.xlsx"
    # df = pd.read_excel(plik_wejsciowy)
    df = pd.read_excel('datadzialy_wide_N_sprzet_o.xlsx')
    
    handel=df[df["wyszczegolnienie"].isin([
    'Handel i naprawy',
    'Handel i naprawa pojazdów',
    'Handel hurtowy',
    'Sprzedaż hurtowa narzędzi technologii informacyjnej i komunikacyjnej',
    'Handel detaliczny'])]
    vector_handel = handel["wyszczegolnienie"].tolist()
    df_transformed_handel = handel.set_index("wyszczegolnienie").T
    df_transformed_handel.columns = vector_handel
    df_transformed_handel.index = [2015, 2016, 2017, 2018, 2019, 2020, 2021,2022]
    correlation_matrix = df_transformed_handel.corr()
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
    plt.title("Macierz korelacji")
    plt.show()
    st.pyplot(fig)

def handel_nak_sprzet_inf_corr():
    st.markdown('### Korelacje - przedsiębiorstwa ponoszące nakłady na sprzęt informatyczny w branżach związanych z handlem')
    
    # plik_wejsciowy =  "C:/Users/Dell/Desktop/streamlit/DATAFRAME/datadzialy_wide_N_sprzet_inf.xlsx"
    # df = pd.read_excel(plik_wejsciowy)
    df = pd.read_excel('datadzialy_wide_N_sprzet_inf.xlsx')
    
    handel=df[df["wyszczegolnienie"].isin([
    'Handel i naprawy',
    'Handel i naprawa pojazdów',
    'Handel hurtowy',
    'Sprzedaż hurtowa narzędzi technologii informacyjnej i komunikacyjnej',
    'Handel detaliczny'])]
    vector_handel = handel["wyszczegolnienie"].tolist()
    df_transformed_handel = handel.set_index("wyszczegolnienie").T
    df_transformed_handel.columns = vector_handel
    df_transformed_handel.index = [2015, 2016, 2017, 2018, 2019, 2020, 2021,2022]
    correlation_matrix = df_transformed_handel.corr()
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
    plt.title("Macierz korelacji")
    plt.show()
    st.pyplot(fig)
    
def handel_nak_sprzet_tele_corr():
    st.markdown('### Korelacje - przedsiębiorstwa ponoszące nakłady na sprzęt telekomunikacyjny w branżach związanych z handlem')
    
    # plik_wejsciowy =  "C:/Users/Dell/Desktop/streamlit/DATAFRAME/datadzialy_wide_N_sprzet_tele.xlsx"
    # df = pd.read_excel(plik_wejsciowy)
    df = pd.read_excel('datadzialy_wide_N_sprzet_tele.xlsx')
    
    handel=df[df["wyszczegolnienie"].isin([
    'Handel i naprawy',
    'Handel i naprawa pojazdów',
    'Handel hurtowy',
    'Sprzedaż hurtowa narzędzi technologii informacyjnej i komunikacyjnej',
    'Handel detaliczny'])]
    vector_handel = handel["wyszczegolnienie"].tolist()
    df_transformed_handel = handel.set_index("wyszczegolnienie").T
    df_transformed_handel.columns = vector_handel
    df_transformed_handel.index = [2015, 2016, 2017, 2018, 2019, 2020, 2021,2022]
    correlation_matrix = df_transformed_handel.corr()
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
    plt.title("Macierz korelacji")
    plt.show()
    st.pyplot(fig)
    
def handel_nak_oprog_corr():
    st.markdown('### Korelacje - przedsiębiorstwa ponoszące nakłady na oprogramowanie w branżach związanych z handlem')
    
    # plik_wejsciowy =  "C:/Users/Dell/Desktop/streamlit/DATAFRAME/datadzialy_wide_N_oprog.xlsx" 
    # df = pd.read_excel(plik_wejsciowy)
    df = pd.read_excel('datadzialy_wide_N_oprog.xlsx')
    
    handel=df[df["wyszczegolnienie"].isin([
    'Handel i naprawy',
    'Handel i naprawa pojazdów',
    'Handel hurtowy',
    'Sprzedaż hurtowa narzędzi technologii informacyjnej i komunikacyjnej',
    'Handel detaliczny'])]
    vector_handel = handel["wyszczegolnienie"].tolist()
    df_transformed_handel = handel.set_index("wyszczegolnienie").T
    df_transformed_handel.columns = vector_handel
    df_transformed_handel.index = [2015, 2016, 2017, 2018, 2019, 2020, 2021,2022]
    correlation_matrix = df_transformed_handel.corr()
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
    plt.title("Macierz korelacji")
    plt.show()
    st.pyplot(fig)
    
def handel_corr_wartosc_nakladow_section():
    st.sidebar.header("Wartość Nakładów")
    pages = {
        "Wartość nakładów - sprzęt/oprogramowanie": handel_wnak_sprzet_opr_corr,
        "Wartość nakładów - sprzęt": handel_wnak_sprzet_corr,
        "Wartość nakładów - sprzęt informatyczny": handel_wnak_sprzet_inf_corr,
        "Wartość nakładów - sprzęt teleinformatyczny": handel_wnak_sprzet_tele_corr,
        "Wartość nakładów - leasing": handel_wnak_leasing_corr,
        "Wartość nakładów - oprogramowanie": handel_wnak_oprog_corr,
    }
    selected_page = st.sidebar.selectbox("Wybierz analizę", pages.keys())
    pages[selected_page]()
    
def handel_wnak_leasing_corr():
    st.markdown('### Korelacje - wartość nakładów poniesionych w przedsiębiorstwach na leasing urządzeń ICT w branżach związanych z handlem')
    
    # plik_wejsciowy =  "C:/Users/Dell/Desktop/streamlit/DATAFRAME/datadzialy_wide_WN_leasing.xlsx"  
    # df = pd.read_excel(plik_wejsciowy)
    df = pd.read_excel('datadzialy_wide_WN_leasing.xlsx')
    
    handel=df[df["wyszczegolnienie"].isin([
    'Handel i naprawy',
    'Handel i naprawa pojazdów',
    'Handel hurtowy',
    'Sprzedaż hurtowa narzędzi technologii informacyjnej i komunikacyjnej',
    'Handel detaliczny'])]
    vector_handel = handel["wyszczegolnienie"].tolist()
    df_transformed_handel = handel.set_index("wyszczegolnienie").T
    df_transformed_handel.columns = vector_handel
    df_transformed_handel.index = [2015, 2016, 2017, 2018, 2019, 2020, 2021,2022]
    correlation_matrix = df_transformed_handel.corr()
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
    plt.title("Macierz korelacji")
    plt.show()
    st.pyplot(fig)
    
def handel_wnak_sprzet_opr_corr():
    st.markdown('### Korelacje - wartość nakładów poniesionych w przedsiębiorstwach na sprzęt i oprogramowanie w branżach związanych z handlem')
    # plik_wejsciowy =  "C:/Users/Dell/Desktop/streamlit/DATAFRAME/datadzialy_wide_WN_sprzet_opr.xlsx" 
    # df = pd.read_excel(plik_wejsciowy)
    df = pd.read_excel('datadzialy_wide_WN_sprzet_opr.xlsx')
    
    handel=df[df["wyszczegolnienie"].isin([
    'Handel i naprawy',
    'Handel i naprawa pojazdów',
    'Handel hurtowy',
    'Sprzedaż hurtowa narzędzi technologii informacyjnej i komunikacyjnej',
    'Handel detaliczny'])]
    vector_handel = handel["wyszczegolnienie"].tolist()
    df_transformed_handel = handel.set_index("wyszczegolnienie").T
    df_transformed_handel.columns = vector_handel
    df_transformed_handel.index = [2015, 2016, 2017, 2018, 2019, 2020, 2021,2022]
    correlation_matrix = df_transformed_handel.corr()
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
    plt.title("Macierz korelacji")
    plt.show()
    st.pyplot(fig)
    
def handel_wnak_sprzet_corr():
    st.markdown('### Korelacje - wartość nakładów poniesionych w przedsiębiorstwach na sprzęt w branżach związanych z handlem')
    
    # plik_wejsciowy =  "C:/Users/Dell/Desktop/streamlit/DATAFRAME/datadzialy_wide_WN_sprzet_o.xlsx"
    # df = pd.read_excel(plik_wejsciowy)
    df = pd.read_excel('datadzialy_wide_WN_sprzet_o.xlsx')
    
    handel=df[df["wyszczegolnienie"].isin([
    'Handel i naprawy',
    'Handel i naprawa pojazdów',
    'Handel hurtowy',
    'Sprzedaż hurtowa narzędzi technologii informacyjnej i komunikacyjnej',
    'Handel detaliczny'])]
    vector_handel = handel["wyszczegolnienie"].tolist()
    df_transformed_handel = handel.set_index("wyszczegolnienie").T
    df_transformed_handel.columns = vector_handel
    df_transformed_handel.index = [2015, 2016, 2017, 2018, 2019, 2020, 2021,2022]
    correlation_matrix = df_transformed_handel.corr()
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
    plt.title("Macierz korelacji")
    plt.show()
    st.pyplot(fig)

def handel_wnak_sprzet_inf_corr():
    st.markdown('### Korelacje - wartość nakładów poniesionych w przedsiębiorstwach na sprzęt informatyczny w branżach związanych z handlem')
    
    # plik_wejsciowy =  "C:/Users/Dell/Desktop/streamlit/DATAFRAME/datadzialy_wide_WN_sprzet_inf.xlsx" 
    # df = pd.read_excel(plik_wejsciowy)
    df = pd.read_excel('datadzialy_wide_WN_sprzet_inf.xlsx')
    
    handel=df[df["wyszczegolnienie"].isin([
    'Handel i naprawy',
    'Handel i naprawa pojazdów',
    'Handel hurtowy',
    'Sprzedaż hurtowa narzędzi technologii informacyjnej i komunikacyjnej',
    'Handel detaliczny'])]
    vector_handel = handel["wyszczegolnienie"].tolist()
    df_transformed_handel = handel.set_index("wyszczegolnienie").T
    df_transformed_handel.columns = vector_handel
    df_transformed_handel.index = [2015, 2016, 2017, 2018, 2019, 2020, 2021,2022]
    correlation_matrix = df_transformed_handel.corr()
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
    plt.title("Macierz korelacji")
    plt.show()
    st.pyplot(fig)
    
def handel_wnak_sprzet_tele_corr():
    st.markdown('### Korelacje - wartość nakładów poniesionych w przedsiębiorstwach na sprzęt telekomunikacyjny w branżach związanych z handlem')
    
    # plik_wejsciowy =  "C:/Users/Dell/Desktop/streamlit/DATAFRAME/datadzialy_wide_WN_sprzet_tele.xlsx"
    # df = pd.read_excel(plik_wejsciowy)
    df = pd.read_excel('datadzialy_wide_WN_sprzet_tele.xlsx')
    
    handel=df[df["wyszczegolnienie"].isin([
    'Handel i naprawy',
    'Handel i naprawa pojazdów',
    'Handel hurtowy',
    'Sprzedaż hurtowa narzędzi technologii informacyjnej i komunikacyjnej',
    'Handel detaliczny'])]
    vector_handel = handel["wyszczegolnienie"].tolist()
    df_transformed_handel = handel.set_index("wyszczegolnienie").T
    df_transformed_handel.columns = vector_handel
    df_transformed_handel.index = [2015, 2016, 2017, 2018, 2019, 2020, 2021,2022]
    correlation_matrix = df_transformed_handel.corr()
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
    plt.title("Macierz korelacji")
    plt.show()
    st.pyplot(fig)
    
def handel_wnak_oprog_corr():
    st.markdown('### Korelacje - wartość nakładów poniesionych w przedsiębiorstwach na oprogramowanie w branżach związanych z handlem')
    
    # plik_wejsciowy =  "C:/Users/Dell/Desktop/streamlit/DATAFRAME/datadzialy_wide_WN_oprog.xlsx" 
    # df = pd.read_excel(plik_wejsciowy)
    df = pd.read_excel('datadzialy_wide_WN_oprog.xlsx')
    
    handel=df[df["wyszczegolnienie"].isin([
    'Handel i naprawy',
    'Handel i naprawa pojazdów',
    'Handel hurtowy',
    'Sprzedaż hurtowa narzędzi technologii informacyjnej i komunikacyjnej',
    'Handel detaliczny'])]
    vector_handel = handel["wyszczegolnienie"].tolist()
    df_transformed_handel = handel.set_index("wyszczegolnienie").T
    df_transformed_handel.columns = vector_handel
    df_transformed_handel.index = [2015, 2016, 2017, 2018, 2019, 2020, 2021,2022]
    correlation_matrix = df_transformed_handel.corr()
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
    plt.title("Macierz korelacji")
    plt.show()
    st.pyplot(fig)
    
    
def handel_corr_pracownicy_section():
    st.sidebar.header("Pracownicy")
    pages = {
        "Przedsiębiorstwa zatrudniające pracowników": handel_przed_o_corr,
        "Przedsiębiorstwa z dostępem do internetu": handel_przed_di_corr,
        "Pracujący ogółem": handel_pr_o_corr,
        "Pracownicy z wyższym wykształceniem": handel_pr_ww_corr,
        "Pracownicy z dostępem do internetu": handel_pr_di_corr,
        "Pracownicy z urządzeniami przenośnymi": handel_pr_up_corr,
    }
    selected_page = st.sidebar.selectbox("Wybierz analizę", pages.keys())
    pages[selected_page]()
    
def handel_przed_o_corr():
    st.markdown('### Korelacje - ilość przedsiębiorstw w branżach związanych z handlem')
    
    # plik_wejsciowy =  "C:/Users/Dell/Desktop/streamlit/DATAFRAME/datadzialy_wide_P_przed_o.xlsx"
    # df = pd.read_excel(plik_wejsciowy)
    df = pd.read_excel('datadzialy_wide_P_przed_o.xlsx')
    
    handel=df[df["wyszczegolnienie"].isin([
    'Handel i naprawy',
    'Handel i naprawa pojazdów',
    'Handel hurtowy',
    'Sprzedaż hurtowa narzędzi technologii informacyjnej i komunikacyjnej',
    'Handel detaliczny'])]
    vector_handel = handel["wyszczegolnienie"].tolist()
    df_transformed_handel = handel.set_index("wyszczegolnienie").T
    df_transformed_handel.columns = vector_handel
    df_transformed_handel.index = [2015, 2016, 2017, 2018, 2019, 2020, 2021]
    correlation_matrix = df_transformed_handel.corr()
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
    plt.title("Macierz korelacji")
    plt.show()
    st.pyplot(fig)
    
def handel_przed_di_corr():
    st.markdown('### Korelacje - ilość przedsiębiorstw z dostępem do internetu w branżach związanych z handlem')
    
    # plik_wejsciowy =  "C:/Users/Dell/Desktop/streamlit/DATAFRAME/datadzialy_wide_P_przed_di.xlsx" 
    # df = pd.read_excel(plik_wejsciowy)
    df = pd.read_excel('datadzialy_wide_P_przed_di.xlsx')
    
    handel=df[df["wyszczegolnienie"].isin([
    'Handel i naprawy',
    'Handel i naprawa pojazdów',
    'Handel hurtowy',
    'Sprzedaż hurtowa narzędzi technologii informacyjnej i komunikacyjnej',
    'Handel detaliczny'])]
    vector_handel = handel["wyszczegolnienie"].tolist()
    df_transformed_handel = handel.set_index("wyszczegolnienie").T
    df_transformed_handel.columns = vector_handel
    df_transformed_handel.index = [2015, 2016, 2017, 2018, 2019, 2020, 2021]
    correlation_matrix = df_transformed_handel.corr()
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
    plt.title("Macierz korelacji")
    plt.show()
    st.pyplot(fig)
    
def handel_pr_o_corr():
    st.markdown('### Korelacje - ilość pracujących ogółem w branżach związanych z handlem')
    
    # plik_wejsciowy =  "C:/Users/Dell/Desktop/streamlit/DATAFRAME/datadzialy_wide_P_pr_o.xlsx" 
    # df = pd.read_excel(plik_wejsciowy)
    df = pd.read_excel('datadzialy_wide_P_pr_o.xlsx')
    
    handel=df[df["wyszczegolnienie"].isin([
    'Handel i naprawy',
    'Handel i naprawa pojazdów',
    'Handel hurtowy',
    'Sprzedaż hurtowa narzędzi technologii informacyjnej i komunikacyjnej',
    'Handel detaliczny'])]
    vector_handel = handel["wyszczegolnienie"].tolist()
    df_transformed_handel = handel.set_index("wyszczegolnienie").T
    df_transformed_handel.columns = vector_handel
    df_transformed_handel.index = [2015, 2016, 2017, 2018, 2019, 2020, 2021]
    correlation_matrix = df_transformed_handel.corr()
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
    plt.title("Macierz korelacji")
    plt.show()
    st.pyplot(fig)
    
def handel_pr_ww_corr():
    st.markdown('### Korelacje - ilość pracowników z wykształceniem wyższym w branżach związanych z handlem')
    
    # plik_wejsciowy =  "C:/Users/Dell/Desktop/streamlit/DATAFRAME/datadzialy_wide_P_pr_ww.xlsx"
    # df = pd.read_excel(plik_wejsciowy)
    df = pd.read_excel('datadzialy_wide_P_pr_ww.xlsx')
    
    handel=df[df["wyszczegolnienie"].isin([
    'Handel i naprawy',
    'Handel i naprawa pojazdów',
    'Handel hurtowy',
    'Sprzedaż hurtowa narzędzi technologii informacyjnej i komunikacyjnej',
    'Handel detaliczny'])]
    vector_handel = handel["wyszczegolnienie"].tolist()
    df_transformed_handel = handel.set_index("wyszczegolnienie").T
    df_transformed_handel.columns = vector_handel
    df_transformed_handel.index = [2015, 2016, 2017, 2018, 2019, 2020, 2021]
    correlation_matrix = df_transformed_handel.corr()
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
    plt.title("Macierz korelacji")
    plt.show()
    st.pyplot(fig)
    
def handel_pr_di_corr():
    st.markdown('### Korelacje - ilość pracowników z dostępem do internetu w branżach związanych z handlem')
    
    # plik_wejsciowy =  "C:/Users/Dell/Desktop/streamlit/DATAFRAME/datadzialy_wide_P_pr_di.xlsx" 
    # df = pd.read_excel(plik_wejsciowy)
    df = pd.read_excel('datadzialy_wide_P_pr_di.xlsx')
    
    handel=df[df["wyszczegolnienie"].isin([
    'Handel i naprawy',
    'Handel i naprawa pojazdów',
    'Handel hurtowy',
    'Sprzedaż hurtowa narzędzi technologii informacyjnej i komunikacyjnej',
    'Handel detaliczny'])]
    vector_handel = handel["wyszczegolnienie"].tolist()
    df_transformed_handel = handel.set_index("wyszczegolnienie").T
    df_transformed_handel.columns = vector_handel
    df_transformed_handel.index = [2015, 2016, 2017, 2018, 2019, 2020, 2021]
    correlation_matrix = df_transformed_handel.corr()
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
    plt.title("Macierz korelacji")
    plt.show()
    st.pyplot(fig)
    
def handel_pr_up_corr():
    st.markdown('### Korelacje - ilość pracowników dysponujących urządzeniami przenośnymi w branżach związanych z handlem')
    
    # plik_wejsciowy =  "C:/Users/Dell/Desktop/streamlit/DATAFRAME/datadzialy_wide_P_pr_up.xlsx" 
    # df = pd.read_excel(plik_wejsciowy)
    df = pd.read_excel('datadzialy_wide_P_pr_up.xlsx')
    
    handel=df[df["wyszczegolnienie"].isin([
    'Handel i naprawy',
    'Handel i naprawa pojazdów',
    'Handel hurtowy',
    'Sprzedaż hurtowa narzędzi technologii informacyjnej i komunikacyjnej',
    'Handel detaliczny'])]
    vector_handel = handel["wyszczegolnienie"].tolist()
    df_transformed_handel = handel.set_index("wyszczegolnienie").T
    df_transformed_handel.columns = vector_handel
    df_transformed_handel.index = [2015, 2016, 2017, 2018, 2019, 2020, 2021]
    correlation_matrix = df_transformed_handel.corr()
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
    plt.title("Macierz korelacji")
    plt.show()
    st.pyplot(fig)
    
def handel_corr_rodzaje_polaczen_section():
    st.sidebar.header("Rodzaje Połączeń")
    pages = {
        "szerokopasmowy  dostęp do Internetu": handel_rodz_pol_szer_corr,
        "szerokopasmowy  dostęp do Internetu poprzez łącze DSL": handel_rodz_pol_szerDSL_corr,
    }
    selected_page = st.sidebar.selectbox("Wybierz analizę", pages.keys())
    pages[selected_page]()
    
def handel_rodz_pol_szer_corr():
    st.markdown('### Korelacje - ilość przedsiębiorstw z dostępem do internetu poprzez łącze szerokopasmowe w branżach związanych z handlem')
    
    # plik_wejsciowy =  "C:/Users/Dell/Desktop/streamlit/DATAFRAME/datadzialy_wide_RP_od_szer_o.xlsx"
    # df = pd.read_excel(plik_wejsciowy)
    df = pd.read_excel('datadzialy_wide_RP_od_szer_o.xlsx')
    
    handel=df[df["wyszczegolnienie"].isin([
    'Handel i naprawy',
    'Handel i naprawa pojazdów',
    'Handel hurtowy',
    'Sprzedaż hurtowa narzędzi technologii informacyjnej i komunikacyjnej',
    'Handel detaliczny'])]
    vector_handel = handel["wyszczegolnienie"].tolist()
    df_transformed_handel = handel.set_index("wyszczegolnienie").T
    df_transformed_handel.columns = vector_handel
    df_transformed_handel.index = [2015, 2016, 2017, 2018, 2019, 2020, 2021,2022]
    correlation_matrix = df_transformed_handel.corr()
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
    plt.title("Macierz korelacji")
    plt.show()
    st.pyplot(fig)
    
def handel_rodz_pol_szerDSL_corr():
    st.markdown('### Korelacje - ilość przedsiębiorstw z dostępem do internetu poprzez łącze szerokopasmowe z użyciem technologii DSL w branżach związanych z handlem')
    
    # plik_wejsciowy =  "C:/Users/Dell/Desktop/streamlit/DATAFRAME/datadzialy_wide_RP_od_szer_DSL.xlsx"
    # df = pd.read_excel(plik_wejsciowy)
    df = pd.read_excel('datadzialy_wide_RP_od_szer_DSL.xlsx')
    
    handel=df[df["wyszczegolnienie"].isin([
    'Handel i naprawy',
    'Handel i naprawa pojazdów',
    'Handel hurtowy',
    'Sprzedaż hurtowa narzędzi technologii informacyjnej i komunikacyjnej',
    'Handel detaliczny'])]
    vector_handel = handel["wyszczegolnienie"].tolist()
    df_transformed_handel = handel.set_index("wyszczegolnienie").T
    df_transformed_handel.columns = vector_handel
    df_transformed_handel.index = [2015, 2016, 2017, 2018, 2019, 2020, 2021,2022]
    correlation_matrix = df_transformed_handel.corr()
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
    plt.title("Macierz korelacji")
    plt.show()
    st.pyplot(fig)
    
def handel_corr_predkosc_polaczen_section():
    st.sidebar.header("Prędkość Połączeń")
    pages = {
        "szerokopasmowy  dostęp do Internetu - prędkość połączenia mniej niż 100 Mbit/s": handel_pred_pol_szer_w_corr,
        "szerokopasmowy  dostęp do Internetu - prędkość połączenia przynajmniej 100 Mbit/s": handel_pred_pol_szer_s_corr
    }
    selected_page = st.sidebar.selectbox("Wybierz analizę", pages.keys())
    pages[selected_page]()
    
def handel_pred_pol_szer_w_corr():
    st.markdown('### Korelacje - prędkość połączenia mniej niż 100 Mbit/s w branżach związanych z handlem')
    
    # plik_wejsciowy =  "C:/Users/Dell/Desktop/streamlit/DATAFRAME/datadzialy_wide_PP_od_szer_w.xlsx" 
    # df = pd.read_excel(plik_wejsciowy)
    df = pd.read_excel('datadzialy_wide_PP_od_szer_w.xlsx')
    
    handel=df[df["wyszczegolnienie"].isin([
    'Handel i naprawy',
    'Handel i naprawa pojazdów',
    'Handel hurtowy',
    'Sprzedaż hurtowa narzędzi technologii informacyjnej i komunikacyjnej',
    'Handel detaliczny'])]
    vector_handel = handel["wyszczegolnienie"].tolist()
    df_transformed_handel = handel.set_index("wyszczegolnienie").T
    df_transformed_handel.columns = vector_handel
    df_transformed_handel.index = [2015, 2016, 2017, 2018, 2019, 2020, 2021,2022]
    correlation_matrix = df_transformed_handel.corr()
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
    plt.title("Macierz korelacji")
    plt.show()
    st.pyplot(fig)
    
def handel_pred_pol_szer_s_corr():
    st.markdown('### Korelacje - prędkość połączenia przynajmniej 100 Mbit/s w branżach związanych z handlem')
    
    # plik_wejsciowy =  "C:/Users/Dell/Desktop/streamlit/DATAFRAME/datadzialy_wide_PP_od_szer_s.xlsx" 
    # df = pd.read_excel(plik_wejsciowy)
    df = pd.read_excel('datadzialy_wide_PP_od_szer_s.xlsx')
    
    handel=df[df["wyszczegolnienie"].isin([
    'Handel i naprawy',
    'Handel i naprawa pojazdów',
    'Handel hurtowy',
    'Sprzedaż hurtowa narzędzi technologii informacyjnej i komunikacyjnej',
    'Handel detaliczny'])]
    vector_handel = handel["wyszczegolnienie"].tolist()
    df_transformed_handel = handel.set_index("wyszczegolnienie").T
    df_transformed_handel.columns = vector_handel
    df_transformed_handel.index = [2015, 2016, 2017, 2018, 2019, 2020, 2021,2022]
    correlation_matrix = df_transformed_handel.corr()
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
    plt.title("Macierz korelacji")
    plt.show()
    st.pyplot(fig)


############################KORELACJE - PRZEMYSŁ#######################################################################################
def korelacje_przemysl_section():
    selected_subsection = st.sidebar.selectbox("Wybierz kategorię", [
        "Nakłady", "Pracownicy", "Wartość Nakładów", "Rodzaje Połączeń", "Prędkość Połączeń"
    ])
    if selected_subsection == "Nakłady":
        przemysl_corr_naklady_section()
    elif selected_subsection == "Pracownicy":
        przemysl_corr_pracownicy_section()
    elif selected_subsection == "Wartość Nakładów":
        przemysl_corr_wartosc_nakladow_section()
    elif selected_subsection == "Rodzaje Połączeń":
        przemysl_corr_rodzaje_polaczen_section()
    elif selected_subsection == "Prędkość Połączeń":
        przemysl_corr_predkosc_polaczen_section()
        
def przemysl_corr_naklady_section():
    st.sidebar.header("Nakłady")
    pages = {
        "Nakłady na sprzęt/oprogramowanie": przemysl_nak_sprzet_opr_corr,
        "Nakłady - sprzęt ogółem": przemysl_nak_sprzet_corr,
        "Nakłady - sprzęt informatyczny": przemysl_nak_sprzet_inf_corr,
        "Nakłady - sprzęt telekomunikacyjny": przemysl_nak_sprzet_tele_corr,
        "Nakłady - leasing": przemysl_nak_leasing_corr,
        "Nakłady - oprogramowanie": przemysl_nak_oprog_corr,
    }
    selected_page = st.sidebar.selectbox("Wybierz analizę", pages.keys())
    pages[selected_page]()
    
def przemysl_nak_leasing_corr():
    st.markdown('### Korelacje - ilość przedsiębiorstw, które poniosły nakłady na leasing urządzeń ICT w branżach związanych z przemysłem')
    
    # plik_wejsciowy =  "C:/Users/Dell/Desktop/streamlit/DATAFRAME/datadzialy_wide_N_leasing.xlsx" 
    # df = pd.read_excel(plik_wejsciowy)
    df = pd.read_excel('datadzialy_wide_N_leasing.xlsx')
    
    przemysl=df[df["wyszczegolnienie"].isin([
    'Przetwórstwo przemysłowe',
    'Przemysł spożywczy',
    'Odzież, tekstylia, wyroby skórzane',
    'Urządzenia elektryczne, produkcja maszyn i urządzeń',
    'Produkcja pojazdów',
    'Transport i gospodarka magazynowa',
    'Badania naukowe i prace rozwojowe'])]
    vector_przemysl = przemysl["wyszczegolnienie"].tolist()
    df_transformed_przemysl = przemysl.set_index("wyszczegolnienie").T
    df_transformed_przemysl.columns = vector_przemysl
    df_transformed_przemysl.index = [2015, 2016, 2017, 2018, 2019, 2020, 2021,2022]
    correlation_matrix = df_transformed_przemysl.corr()
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
    plt.title("Macierz korelacji")
    plt.show()
    st.pyplot(fig)
    
def przemysl_nak_sprzet_opr_corr():
    st.markdown('### Korelacje - ilość przedsiębiorstw, które poniosły nakłady na sprzęt i oprogramowanie w branżach związanych z przemysłem')
    
    # plik_wejsciowy =  "C:/Users/Dell/Desktop/streamlit/DATAFRAME/datadzialy_wide_N_sprzet_opr.xlsx" 
    # df = pd.read_excel(plik_wejsciowy)
    df = pd.read_excel('datadzialy_wide_N_sprzet_opr.xlsx')
    
    przemysl=df[df["wyszczegolnienie"].isin([
    'Przetwórstwo przemysłowe',
    'Przemysł spożywczy',
    'Odzież, tekstylia, wyroby skórzane',
    'Urządzenia elektryczne, produkcja maszyn i urządzeń',
    'Produkcja pojazdów',
    'Transport i gospodarka magazynowa',
    'Badania naukowe i prace rozwojowe'])]
    vector_przemysl = przemysl["wyszczegolnienie"].tolist()
    df_transformed_przemysl = przemysl.set_index("wyszczegolnienie").T
    df_transformed_przemysl.columns = vector_przemysl
    df_transformed_przemysl.index = [2015, 2016, 2017, 2018, 2019, 2020, 2021,2022]
    correlation_matrix = df_transformed_przemysl.corr()
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
    plt.title("Macierz korelacji")
    plt.show()
    st.pyplot(fig)
    
def przemysl_nak_sprzet_corr():
    st.markdown('### Korelacje - Korelacje - ilość przedsiębiorstw, które poniosły nakłady na sprzęt w branżach związanych z przemysłem')
    
    # plik_wejsciowy =  "C:/Users/Dell/Desktop/streamlit/DATAFRAME/datadzialy_wide_N_sprzet_o.xlsx"
    # df = pd.read_excel(plik_wejsciowy)
    df = pd.read_excel('datadzialy_wide_N_sprzet_o.xlsx')
    
    przemysl=df[df["wyszczegolnienie"].isin([
    'Przetwórstwo przemysłowe',
    'Przemysł spożywczy',
    'Odzież, tekstylia, wyroby skórzane',
    'Urządzenia elektryczne, produkcja maszyn i urządzeń',
    'Produkcja pojazdów',
    'Transport i gospodarka magazynowa',
    'Badania naukowe i prace rozwojowe'])]
    vector_przemysl = przemysl["wyszczegolnienie"].tolist()
    df_transformed_przemysl = przemysl.set_index("wyszczegolnienie").T
    df_transformed_przemysl.columns = vector_przemysl
    df_transformed_przemysl.index = [2015, 2016, 2017, 2018, 2019, 2020, 2021,2022]
    correlation_matrix = df_transformed_przemysl.corr()
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
    plt.title("Macierz korelacji")
    plt.show()
    st.pyplot(fig)

def przemysl_nak_sprzet_inf_corr():
    st.markdown('### Korelacje - ilość przedsiębiorstw, które poniosły nakłady na sprzęt informatyczny w branżach związanych z przemysłem')
    
    # plik_wejsciowy =  "C:/Users/Dell/Desktop/streamlit/DATAFRAME/datadzialy_wide_N_sprzet_inf.xlsx"
    # df = pd.read_excel(plik_wejsciowy)
    df = pd.read_excel('datadzialy_wide_N_sprzet_inf.xlsx')
    
    przemysl=df[df["wyszczegolnienie"].isin([
    'Przetwórstwo przemysłowe',
    'Przemysł spożywczy',
    'Odzież, tekstylia, wyroby skórzane',
    'Urządzenia elektryczne, produkcja maszyn i urządzeń',
    'Produkcja pojazdów',
    'Transport i gospodarka magazynowa',
    'Badania naukowe i prace rozwojowe'])]
    vector_przemysl = przemysl["wyszczegolnienie"].tolist()
    df_transformed_przemysl = przemysl.set_index("wyszczegolnienie").T
    df_transformed_przemysl.columns = vector_przemysl
    df_transformed_przemysl.index = [2015, 2016, 2017, 2018, 2019, 2020, 2021,2022]
    correlation_matrix = df_transformed_przemysl.corr()
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
    plt.title("Macierz korelacji")
    plt.show()
    st.pyplot(fig)
    
def przemysl_nak_sprzet_tele_corr():
    st.markdown('### Korelacje - ilość przedsiębiorstw, które poniosły nakłady na sprzęt telekomunikacyjny w branżach związanych z przemysłem')
    
    # plik_wejsciowy =  "C:/Users/Dell/Desktop/streamlit/DATAFRAME/datadzialy_wide_N_sprzet_tele.xlsx"
    # df = pd.read_excel(plik_wejsciowy)
    df = pd.read_excel('datadzialy_wide_N_sprzet_tele.xlsx')
    
    przemysl=df[df["wyszczegolnienie"].isin([
    'Przetwórstwo przemysłowe',
    'Przemysł spożywczy',
    'Odzież, tekstylia, wyroby skórzane',
    'Urządzenia elektryczne, produkcja maszyn i urządzeń',
    'Produkcja pojazdów',
    'Transport i gospodarka magazynowa',
    'Badania naukowe i prace rozwojowe'])]
    vector_przemysl = przemysl["wyszczegolnienie"].tolist()
    df_transformed_przemysl = przemysl.set_index("wyszczegolnienie").T
    df_transformed_przemysl.columns = vector_przemysl
    df_transformed_przemysl.index = [2015, 2016, 2017, 2018, 2019, 2020, 2021,2022]
    correlation_matrix = df_transformed_przemysl.corr()
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
    plt.title("Macierz korelacji")
    plt.show()
    st.pyplot(fig)    
    
def przemysl_nak_oprog_corr():
    st.markdown('### Korelacje - ilość przedsiębiorstw, które poniosły nakłady na oprogramowanie w branżach związanych z przemysłem')
    
    # plik_wejsciowy =  "C:/Users/Dell/Desktop/streamlit/DATAFRAME/datadzialy_wide_N_oprog.xlsx"
    # df = pd.read_excel(plik_wejsciowy)
    df = pd.read_excel('datadzialy_wide_N_oprog.xlsx')
    
    przemysl=df[df["wyszczegolnienie"].isin([
    'Przetwórstwo przemysłowe',
    'Przemysł spożywczy',
    'Odzież, tekstylia, wyroby skórzane',
    'Urządzenia elektryczne, produkcja maszyn i urządzeń',
    'Produkcja pojazdów',
    'Transport i gospodarka magazynowa',
    'Badania naukowe i prace rozwojowe'])]
    vector_przemysl = przemysl["wyszczegolnienie"].tolist()
    df_transformed_przemysl = przemysl.set_index("wyszczegolnienie").T
    df_transformed_przemysl.columns = vector_przemysl
    df_transformed_przemysl.index = [2015, 2016, 2017, 2018, 2019, 2020, 2021,2022]
    correlation_matrix = df_transformed_przemysl.corr()
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
    plt.title("Macierz korelacji")
    plt.show()
    st.pyplot(fig)
    
def przemysl_corr_wartosc_nakladow_section():
    st.sidebar.header("Wartość Nakładów")
    pages = {
        "Wartość nakładów - sprzęt/oprogramowanie": przemysl_wnak_sprzet_opr_corr,
        "Wartość nakładów - sprzęt": przemysl_wnak_sprzet_corr,
        "Wartość nakładów - sprzęt informatyczny": przemysl_wnak_sprzet_inf_corr,
        "Wartość nakładów - sprzęt teleinformatyczny": przemysl_wnak_sprzet_tele_corr,
        "Wartość nakładów - leasing": przemysl_wnak_leasing_corr,
        "Wartość nakładów - oprogramowanie": przemysl_wnak_oprog_corr,
    }
    selected_page = st.sidebar.selectbox("Wybierz analizę", pages.keys())
    pages[selected_page]()
    
def przemysl_wnak_leasing_corr():
    st.markdown('### Korelacje - wartość nakładów na leasing urządzeń ICT poniesiona przez przedsiębiorstwa w branżach związanych z przemysłem')
    
    # plik_wejsciowy =  "C:/Users/Dell/Desktop/streamlit/DATAFRAME/datadzialy_wide_WN_leasing.xlsx" 
    # df = pd.read_excel(plik_wejsciowy)
    df = pd.read_excel('datadzialy_wide_WN_leasing.xlsx')
    
    przemysl=df[df["wyszczegolnienie"].isin([
    'Przetwórstwo przemysłowe',
    'Przemysł spożywczy',
    'Odzież, tekstylia, wyroby skórzane',
    'Urządzenia elektryczne, produkcja maszyn i urządzeń',
    'Produkcja pojazdów',
    'Transport i gospodarka magazynowa',
    'Badania naukowe i prace rozwojowe'])]
    vector_przemysl = przemysl["wyszczegolnienie"].tolist()
    df_transformed_przemysl = przemysl.set_index("wyszczegolnienie").T
    df_transformed_przemysl.columns = vector_przemysl
    df_transformed_przemysl.index = [2015, 2016, 2017, 2018, 2019, 2020, 2021,2022]
    correlation_matrix = df_transformed_przemysl.corr()
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
    plt.title("Macierz korelacji")
    plt.show()
    st.pyplot(fig)
    
def przemysl_wnak_sprzet_opr_corr():
    st.markdown('### Korelacje - wartość nakładów na sprzęt i oprogramowanie poniesiona przez przedsiębiorstwa w branżach związanych z przemysłem')
    
    # plik_wejsciowy =  "C:/Users/Dell/Desktop/streamlit/DATAFRAME/datadzialy_wide_WN_sprzet_opr.xlsx" 
    # df = pd.read_excel(plik_wejsciowy)
    df = pd.read_excel('datadzialy_wide_WN_sprzet_opr.xlsx')
    
    przemysl=df[df["wyszczegolnienie"].isin([
    'Przetwórstwo przemysłowe',
    'Przemysł spożywczy',
    'Odzież, tekstylia, wyroby skórzane',
    'Urządzenia elektryczne, produkcja maszyn i urządzeń',
    'Produkcja pojazdów',
    'Transport i gospodarka magazynowa',
    'Badania naukowe i prace rozwojowe'])]
    vector_przemysl = przemysl["wyszczegolnienie"].tolist()
    df_transformed_przemysl = przemysl.set_index("wyszczegolnienie").T
    df_transformed_przemysl.columns = vector_przemysl
    df_transformed_przemysl.index = [2015, 2016, 2017, 2018, 2019, 2020, 2021,2022]
    correlation_matrix = df_transformed_przemysl.corr()
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
    plt.title("Macierz korelacji")
    plt.show()
    st.pyplot(fig)
    
def przemysl_wnak_sprzet_corr():
    st.markdown('### Korelacje - wartość nakładów na sprzęt poniesiona przez przedsiębiorstwa w branżach związanych z przemysłem')
    
    # plik_wejsciowy =  "C:/Users/Dell/Desktop/streamlit/DATAFRAME/datadzialy_wide_WN_sprzet_o.xlsx" 
    # df = pd.read_excel(plik_wejsciowy)
    df = pd.read_excel('datadzialy_wide_WN_sprzet_o.xlsx')
    
    przemysl=df[df["wyszczegolnienie"].isin([
    'Przetwórstwo przemysłowe',
    'Przemysł spożywczy',
    'Odzież, tekstylia, wyroby skórzane',
    'Urządzenia elektryczne, produkcja maszyn i urządzeń',
    'Produkcja pojazdów',
    'Transport i gospodarka magazynowa',
    'Badania naukowe i prace rozwojowe'])]
    vector_przemysl = przemysl["wyszczegolnienie"].tolist()
    df_transformed_przemysl = przemysl.set_index("wyszczegolnienie").T
    df_transformed_przemysl.columns = vector_przemysl
    df_transformed_przemysl.index = [2015, 2016, 2017, 2018, 2019, 2020, 2021,2022]
    correlation_matrix = df_transformed_przemysl.corr()
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
    plt.title("Macierz korelacji")
    plt.show()
    st.pyplot(fig)

def przemysl_wnak_sprzet_inf_corr():
    st.markdown('### Korelacje - wartość nakładów na sprzęt informatyczny poniesiona przez przedsiębiorstwa w branżach związanych z przemysłem')
    
    # plik_wejsciowy =  "C:/Users/Dell/Desktop/streamlit/DATAFRAME/datadzialy_wide_WN_sprzet_inf.xlsx"
    # df = pd.read_excel(plik_wejsciowy)
    df = pd.read_excel('datadzialy_wide_WN_sprzet_inf.xlsx')
    
    przemysl=df[df["wyszczegolnienie"].isin([
    'Przetwórstwo przemysłowe',
    'Przemysł spożywczy',
    'Odzież, tekstylia, wyroby skórzane',
    'Urządzenia elektryczne, produkcja maszyn i urządzeń',
    'Produkcja pojazdów',
    'Transport i gospodarka magazynowa',
    'Badania naukowe i prace rozwojowe'])]
    vector_przemysl = przemysl["wyszczegolnienie"].tolist()
    df_transformed_przemysl = przemysl.set_index("wyszczegolnienie").T
    df_transformed_przemysl.columns = vector_przemysl
    df_transformed_przemysl.index = [2015, 2016, 2017, 2018, 2019, 2020, 2021,2022]
    correlation_matrix = df_transformed_przemysl.corr()
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
    plt.title("Macierz korelacji")
    plt.show()
    st.pyplot(fig)
    
def przemysl_wnak_sprzet_tele_corr():
    st.markdown('### Korelacje - wartość nakładów na sprzęt telekomunikacyjny poniesiona przez przedsiębiorstwa w branżach związanych z przemysłem')
    
    # plik_wejsciowy =  "C:/Users/Dell/Desktop/streamlit/DATAFRAME/datadzialy_wide_WN_sprzet_tele.xlsx"
    # df = pd.read_excel(plik_wejsciowy)
    df = pd.read_excel('datadzialy_wide_WN_sprzet_tele.xlsx')
    
    przemysl=df[df["wyszczegolnienie"].isin([
    'Przetwórstwo przemysłowe',
    'Przemysł spożywczy',
    'Odzież, tekstylia, wyroby skórzane',
    'Urządzenia elektryczne, produkcja maszyn i urządzeń',
    'Produkcja pojazdów',
    'Transport i gospodarka magazynowa',
    'Badania naukowe i prace rozwojowe'])]
    vector_przemysl = przemysl["wyszczegolnienie"].tolist()
    df_transformed_przemysl = przemysl.set_index("wyszczegolnienie").T
    df_transformed_przemysl.columns = vector_przemysl
    df_transformed_przemysl.index = [2015, 2016, 2017, 2018, 2019, 2020, 2021,2022]
    correlation_matrix = df_transformed_przemysl.corr()
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
    plt.title("Macierz korelacji")
    plt.show()
    st.pyplot(fig)
    
def przemysl_wnak_oprog_corr():
    st.markdown('### Korelacje - wartość nakładów na oprogramowanie poniesiona przez przedsiębiorstwa w branżach związanych z przemysłem')
    
    # plik_wejsciowy =  "C:/Users/Dell/Desktop/streamlit/DATAFRAME/datadzialy_wide_WN_oprog.xlsx" 
    # df = pd.read_excel(plik_wejsciowy)
    df = pd.read_excel('datadzialy_wide_WN_oprog.xlsx')
    
    przemysl=df[df["wyszczegolnienie"].isin([
    'Przetwórstwo przemysłowe',
    'Przemysł spożywczy',
    'Odzież, tekstylia, wyroby skórzane',
    'Urządzenia elektryczne, produkcja maszyn i urządzeń',
    'Produkcja pojazdów',
    'Transport i gospodarka magazynowa',
    'Badania naukowe i prace rozwojowe'])]
    vector_przemysl = przemysl["wyszczegolnienie"].tolist()
    df_transformed_przemysl = przemysl.set_index("wyszczegolnienie").T
    df_transformed_przemysl.columns = vector_przemysl
    df_transformed_przemysl.index = [2015, 2016, 2017, 2018, 2019, 2020, 2021,2022]
    correlation_matrix = df_transformed_przemysl.corr()
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
    plt.title("Macierz korelacji")
    plt.show()
    st.pyplot(fig)
    
def przemysl_corr_pracownicy_section():
    st.sidebar.header("Pracownicy")
    pages = {
        "Przedsiębiorstwa zatrudniające pracowników": przemysl_przed_o_corr,
        "Przedsiębiorstwa z dostępem do internetu": przemysl_przed_di_corr,
        "Pracujący ogółem": przemysl_pr_o_corr,
        "Pracownicy z wyższym wykształceniem": przemysl_pr_ww_corr,
        "Pracownicy z dostępem do internetu": przemysl_pr_di_corr,
        "Pracownicy z urządzeniami przenośnymi": przemysl_pr_up_corr,
    }
    selected_page = st.sidebar.selectbox("Wybierz analizę", pages.keys())
    pages[selected_page]()
    
def przemysl_przed_o_corr():
    st.markdown('### Korelacje - ilość przedsiębiorstw w branżach związanych z przemysłem')
    
    # plik_wejsciowy =  "C:/Users/Dell/Desktop/streamlit/DATAFRAME/datadzialy_wide_P_przed_o.xlsx"
    # df = pd.read_excel(plik_wejsciowy)
    df = pd.read_excel('datadzialy_wide_P_przed_o.xlsx')
    
    przemysl=df[df["wyszczegolnienie"].isin([
    'Przetwórstwo przemysłowe',
    'Przemysł spożywczy',
    'Odzież, tekstylia, wyroby skórzane',
    'Urządzenia elektryczne, produkcja maszyn i urządzeń',
    'Produkcja pojazdów',
    'Transport i gospodarka magazynowa',
    'Badania naukowe i prace rozwojowe'])]
    vector_przemysl = przemysl["wyszczegolnienie"].tolist()
    df_transformed_przemysl = przemysl.set_index("wyszczegolnienie").T
    df_transformed_przemysl.columns = vector_przemysl
    df_transformed_przemysl.index = [2015, 2016, 2017, 2018, 2019, 2020, 2021]
    correlation_matrix = df_transformed_przemysl.corr()
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
    plt.title("Macierz korelacji")
    plt.show()
    st.pyplot(fig)
    
def przemysl_przed_di_corr():
    st.markdown('### Korelacje - ilość przedsiębiorstw z dostępem do internetu w branżach związanych z przemysłem')
    
    # plik_wejsciowy =  "C:/Users/Dell/Desktop/streamlit/DATAFRAME/datadzialy_wide_P_przed_di.xlsx" 
    # df = pd.read_excel(plik_wejsciowy)
    df = pd.read_excel('datadzialy_wide_P_przed_di.xlsx')

    przemysl=df[df["wyszczegolnienie"].isin([
    'Przetwórstwo przemysłowe',
    'Przemysł spożywczy',
    'Odzież, tekstylia, wyroby skórzane',
    'Urządzenia elektryczne, produkcja maszyn i urządzeń',
    'Produkcja pojazdów',
    'Transport i gospodarka magazynowa',
    'Badania naukowe i prace rozwojowe'])]
    vector_przemysl = przemysl["wyszczegolnienie"].tolist()
    df_transformed_przemysl = przemysl.set_index("wyszczegolnienie").T
    df_transformed_przemysl.columns = vector_przemysl
    df_transformed_przemysl.index = [2015, 2016, 2017, 2018, 2019, 2020, 2021]
    correlation_matrix = df_transformed_przemysl.corr()
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
    plt.title("Macierz korelacji")
    plt.show()
    st.pyplot(fig)
    
def przemysl_pr_o_corr():
    st.markdown('### Korelacje - ilość pracujących ogółem w branżach związanych z przemysłem')
    
    # plik_wejsciowy =  "C:/Users/Dell/Desktop/streamlit/DATAFRAME/datadzialy_wide_P_pr_o.xlsx" 
    # df = pd.read_excel(plik_wejsciowy)
    df = pd.read_excel('datadzialy_wide_P_pr_o.xlsx')
    
    przemysl=df[df["wyszczegolnienie"].isin([
    'Przetwórstwo przemysłowe',
    'Przemysł spożywczy',
    'Odzież, tekstylia, wyroby skórzane',
    'Urządzenia elektryczne, produkcja maszyn i urządzeń',
    'Produkcja pojazdów',
    'Transport i gospodarka magazynowa',
    'Badania naukowe i prace rozwojowe'])]
    vector_przemysl = przemysl["wyszczegolnienie"].tolist()
    df_transformed_przemysl = przemysl.set_index("wyszczegolnienie").T
    df_transformed_przemysl.columns = vector_przemysl
    df_transformed_przemysl.index = [2015, 2016, 2017, 2018, 2019, 2020, 2021]
    correlation_matrix = df_transformed_przemysl.corr()
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
    plt.title("Macierz korelacji")
    plt.show()
    st.pyplot(fig)
    
def przemysl_pr_ww_corr():
    st.markdown('### Korelacje - ilość pracowników z wyższym wykształceniem w branżach związanych z przemysłem')
    
    # plik_wejsciowy =  "C:/Users/Dell/Desktop/streamlit/DATAFRAME/datadzialy_wide_P_pr_ww.xlsx" 
    # df = pd.read_excel(plik_wejsciowy)
    df = pd.read_excel('datadzialy_wide_P_pr_ww.xlsx')
    
    przemysl=df[df["wyszczegolnienie"].isin([
    'Przetwórstwo przemysłowe',
    'Przemysł spożywczy',
    'Odzież, tekstylia, wyroby skórzane',
    'Urządzenia elektryczne, produkcja maszyn i urządzeń',
    'Produkcja pojazdów',
    'Transport i gospodarka magazynowa',
    'Badania naukowe i prace rozwojowe'])]
    vector_przemysl = przemysl["wyszczegolnienie"].tolist()
    df_transformed_przemysl = przemysl.set_index("wyszczegolnienie").T
    df_transformed_przemysl.columns = vector_przemysl
    df_transformed_przemysl.index = [2015, 2016, 2017, 2018, 2019, 2020, 2021]
    correlation_matrix = df_transformed_przemysl.corr()
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
    plt.title("Macierz korelacji")
    plt.show()
    st.pyplot(fig)
    
def przemysl_pr_di_corr():
    st.markdown('### Korelacje - ilość pracowników z dostępem do internetu w branżach związanych z przemysłem')
    
    # plik_wejsciowy =  "C:/Users/Dell/Desktop/streamlit/DATAFRAME/datadzialy_wide_P_pr_di.xlsx" 
    # df = pd.read_excel(plik_wejsciowy)
    df = pd.read_excel('datadzialy_wide_P_pr_di.xlsx')
    
    przemysl=df[df["wyszczegolnienie"].isin([
    'Przetwórstwo przemysłowe',
    'Przemysł spożywczy',
    'Odzież, tekstylia, wyroby skórzane',
    'Urządzenia elektryczne, produkcja maszyn i urządzeń',
    'Produkcja pojazdów',
    'Transport i gospodarka magazynowa',
    'Badania naukowe i prace rozwojowe'])]
    vector_przemysl = przemysl["wyszczegolnienie"].tolist()
    df_transformed_przemysl = przemysl.set_index("wyszczegolnienie").T
    df_transformed_przemysl.columns = vector_przemysl
    df_transformed_przemysl.index = [2015, 2016, 2017, 2018, 2019, 2020, 2021]
    correlation_matrix = df_transformed_przemysl.corr()
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
    plt.title("Macierz korelacji")
    plt.show()
    st.pyplot(fig)
    
def przemysl_pr_up_corr():
    st.markdown('### Korelacje - ilość pracowników dysponujących urządzeniami przenośnymi w branżach związanych z przemysłem')
    
    # plik_wejsciowy =  "C:/Users/Dell/Desktop/streamlit/DATAFRAME/datadzialy_wide_P_pr_up.xlsx"
    # df = pd.read_excel(plik_wejsciowy)
    df = pd.read_excel('datadzialy_wide_P_pr_up.xlsx')
    
    przemysl=df[df["wyszczegolnienie"].isin([
    'Przetwórstwo przemysłowe',
    'Przemysł spożywczy',
    'Odzież, tekstylia, wyroby skórzane',
    'Urządzenia elektryczne, produkcja maszyn i urządzeń',
    'Produkcja pojazdów',
    'Transport i gospodarka magazynowa',
    'Badania naukowe i prace rozwojowe'])]
    vector_przemysl = przemysl["wyszczegolnienie"].tolist()
    df_transformed_przemysl = przemysl.set_index("wyszczegolnienie").T
    df_transformed_przemysl.columns = vector_przemysl
    df_transformed_przemysl.index = [2015, 2016, 2017, 2018, 2019, 2020, 2021]
    correlation_matrix = df_transformed_przemysl.corr()
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
    plt.title("Macierz korelacji")
    plt.show()
    st.pyplot(fig)
    
def przemysl_corr_rodzaje_polaczen_section():
    st.sidebar.header("Rodzaje Połączeń")
    pages = {
        "szerokopasmowy  dostęp do Internetu": przemysl_rodz_pol_szer_corr,
        "szerokopasmowy  dostęp do Internetu poprzez łącze DSL": przemysl_rodz_pol_szerDSL_corr,
    }
    selected_page = st.sidebar.selectbox("Wybierz analizę", pages.keys())
    pages[selected_page]()
    
def przemysl_rodz_pol_szer_corr():
    st.markdown('### Korelacje - ilość przedsiębiorstw z dostępem do internetu poprzez łącze szerokopasmowe w branżach związanych z przemysłem')
    
    # plik_wejsciowy =  "C:/Users/Dell/Desktop/streamlit/DATAFRAME/datadzialy_wide_RP_od_szer_o.xlsx" 
    # df = pd.read_excel(plik_wejsciowy)
    df = pd.read_excel('datadzialy_wide_RP_od_szer_o.xlsx')
    
    przemysl=df[df["wyszczegolnienie"].isin([
    'Przetwórstwo przemysłowe',
    'Przemysł spożywczy',
    'Odzież, tekstylia, wyroby skórzane',
    'Urządzenia elektryczne, produkcja maszyn i urządzeń',
    'Produkcja pojazdów',
    'Transport i gospodarka magazynowa',
    'Badania naukowe i prace rozwojowe'])]
    vector_przemysl = przemysl["wyszczegolnienie"].tolist()
    df_transformed_przemysl = przemysl.set_index("wyszczegolnienie").T
    df_transformed_przemysl.columns = vector_przemysl
    df_transformed_przemysl.index = [2015, 2016, 2017, 2018, 2019, 2020, 2021,2022]
    correlation_matrix = df_transformed_przemysl.corr()
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
    plt.title("Macierz korelacji")
    plt.show()
    st.pyplot(fig)
    
def przemysl_rodz_pol_szerDSL_corr():
    st.markdown('### Korelacje - ilość przedsiębiorstw z dostępem do internetu poprzez łącze szerokopasmowe wykorzystywane z użyciem technologii DSL w branżach związanych z przemysłem')
    
    # plik_wejsciowy =  "C:/Users/Dell/Desktop/streamlit/DATAFRAME/datadzialy_wide_RP_od_szer_DSL.xlsx" 
    # df = pd.read_excel(plik_wejsciowy)
    df = pd.read_excel('datadzialy_wide_RP_od_szer_DSL.xlsx')
    
    przemysl=df[df["wyszczegolnienie"].isin([
    'Przetwórstwo przemysłowe',
    'Przemysł spożywczy',
    'Odzież, tekstylia, wyroby skórzane',
    'Urządzenia elektryczne, produkcja maszyn i urządzeń',
    'Produkcja pojazdów',
    'Transport i gospodarka magazynowa',
    'Badania naukowe i prace rozwojowe'])]
    vector_przemysl = przemysl["wyszczegolnienie"].tolist()
    df_transformed_przemysl = przemysl.set_index("wyszczegolnienie").T
    df_transformed_przemysl.columns = vector_przemysl
    df_transformed_przemysl.index = [2015, 2016, 2017, 2018, 2019, 2020, 2021,2022]
    correlation_matrix = df_transformed_przemysl.corr()
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
    plt.title("Macierz korelacji")
    plt.show()
    st.pyplot(fig)
    
def przemysl_corr_predkosc_polaczen_section():
    st.sidebar.header("Prędkość Połączeń")
    pages = {
        "szerokopasmowy  dostęp do Internetu - prędkość połączenia mniej niż 100 Mbit/s": przemysl_pred_pol_szer_w_corr,
        "szerokopasmowy  dostęp do Internetu - prędkość połączenia przynajmniej 100 Mbit/s": przemysl_pred_pol_szer_s_corr
    }
    selected_page = st.sidebar.selectbox("Wybierz analizę", pages.keys())
    pages[selected_page]()
    
def przemysl_pred_pol_szer_w_corr():
    st.markdown('### Korelacje - prędkość połączenia mniej niż 100 Mbit/s w przedsiębiorstwach związanych z przemysłem')
    
    # plik_wejsciowy =  "C:/Users/Dell/Desktop/streamlit/DATAFRAME/datadzialy_wide_PP_od_szer_w.xlsx"
    # df = pd.read_excel(plik_wejsciowy)
    df = pd.read_excel('datadzialy_wide_PP_od_szer_w.xlsx')
    
    przemysl=df[df["wyszczegolnienie"].isin([
    'Przetwórstwo przemysłowe',
    'Przemysł spożywczy',
    'Odzież, tekstylia, wyroby skórzane',
    'Urządzenia elektryczne, produkcja maszyn i urządzeń',
    'Produkcja pojazdów',
    'Transport i gospodarka magazynowa',
    'Badania naukowe i prace rozwojowe'])]
    vector_przemysl = przemysl["wyszczegolnienie"].tolist()
    df_transformed_przemysl = przemysl.set_index("wyszczegolnienie").T
    df_transformed_przemysl.columns = vector_przemysl
    df_transformed_przemysl.index = [2015, 2016, 2017, 2018, 2019, 2020, 2021,2022]
    correlation_matrix = df_transformed_przemysl.corr()
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
    plt.title("Macierz korelacji")
    plt.show()
    st.pyplot(fig)
    
def przemysl_pred_pol_szer_s_corr():
    st.markdown('### Korelacje - prędkość połączenia przynajmniej 100 Mbit/s w przedsiębiorstwach związanych z przemysłem')
    
    # plik_wejsciowy =  "C:/Users/Dell/Desktop/streamlit/DATAFRAME/datadzialy_wide_PP_od_szer_s.xlsx"
    # df = pd.read_excel(plik_wejsciowy)
    df = pd.read_excel('datadzialy_wide_PP_od_szer_s.xlsx')
    
    przemysl=df[df["wyszczegolnienie"].isin([
    'Przetwórstwo przemysłowe',
    'Przemysł spożywczy',
    'Odzież, tekstylia, wyroby skórzane',
    'Urządzenia elektryczne, produkcja maszyn i urządzeń',
    'Produkcja pojazdów',
    'Transport i gospodarka magazynowa',
    'Badania naukowe i prace rozwojowe'])]
    vector_przemysl = przemysl["wyszczegolnienie"].tolist()
    df_transformed_przemysl = przemysl.set_index("wyszczegolnienie").T
    df_transformed_przemysl.columns = vector_przemysl
    df_transformed_przemysl.index = [2015, 2016, 2017, 2018, 2019, 2020, 2021,2022]
    correlation_matrix = df_transformed_przemysl.corr()
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
    plt.title("Macierz korelacji")
    plt.show()
    st.pyplot(fig)

############################KORELACJE - BUDOWNICTWO#####################################################################################
def korelacje_bud_section():
    selected_subsection = st.sidebar.selectbox("Wybierz kategorię", [
        "Nakłady", "Pracownicy", "Wartość Nakładów", "Rodzaje Połączeń", "Prędkość Połączeń"
    ])
    if selected_subsection == "Nakłady":
        budcorr_naklady_section()
    elif selected_subsection == "Pracownicy":
        budcorr_pracownicy_section()
    elif selected_subsection == "Wartość Nakładów":
        budcorr_wartosc_nakladow_section()
    elif selected_subsection == "Rodzaje Połączeń":
        budcorr_rodzaje_polaczen_section()
    elif selected_subsection == "Prędkość Połączeń":
        budcorr_predkosc_polaczen_section()
        
def budcorr_naklady_section():
    st.sidebar.header("Nakłady")
    pages = {
        "Nakłady na sprzęt/oprogramowanie": budnak_sprzet_opr_corr,
        "Nakłady - sprzęt ogółem": budnak_sprzet_corr,
        "Nakłady - sprzęt informatyczny": budnak_sprzet_inf_corr,
        "Nakłady - sprzęt telekomunikacyjny": budnak_sprzet_tele_corr,
        "Nakłady - leasing": budnak_leasing_corr,
        "Nakłady - oprogramowanie": budnak_oprog_corr,
    }
    selected_page = st.sidebar.selectbox("Wybierz analizę", pages.keys())
    pages[selected_page]()
    
def budnak_leasing_corr():
    st.markdown('### Korelacje - przedsiębiorstwa, które poniosły nakłady na leasing urządzeń ICT w branżach związanych z budownictwem')
    
    # plik_wejsciowy =  "C:/Users/Dell/Desktop/streamlit/DATAFRAME/datadzialy_wide_N_leasing.xlsx"
    # df = pd.read_excel(plik_wejsciowy)
    df = pd.read_excel('datadzialy_wide_N_leasing.xlsx')
    
    budownictwo=df[df["wyszczegolnienie"].isin(['Wyroby drewniane, papiernictwo i poligrafia',
'Przemysł chemiczny i farmaceutyczny, ceramika i przetwarzanie materiałów niemetalicznych',
'Produkcja metali i metalowych wyrobów gotowych',
'Energia elektryczna, gaz, ciepło',
'Dostawa wody, ścieki i odpady',
'Budownictwo',
'Transport i gospodarka magazynowa',
'Obsługa rynku nieruchomości',
'Administrowanie i działalność wspierająca',
'Wynajem, dzierżawa, doradztwo personalne',
'Ochrona, utrzymanie czystości itp.'])]
    vector_budownictwo = budownictwo["wyszczegolnienie"].tolist()
    df_transformed_bud = budownictwo.set_index("wyszczegolnienie").T
    df_transformed_bud.columns = vector_budownictwo
    df_transformed_bud.index = [2015, 2016, 2017, 2018, 2019, 2020, 2021,2022]
    correlation_matrix = df_transformed_bud.corr()
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
    plt.title("Macierz korelacji")
    plt.show()
    st.pyplot(fig)
    
def budnak_sprzet_opr_corr():
    st.markdown('### Korelacje - przedsiębiorstwa, które poniosły nakłady na sprzęt i oprogramowanie w branżach związanych z budownictwem')
    
    # plik_wejsciowy =  "C:/Users/Dell/Desktop/streamlit/DATAFRAME/datadzialy_wide_N_sprzet_opr.xlsx"
    # df = pd.read_excel(plik_wejsciowy)
    df = pd.read_excel('datadzialy_wide_N_sprzet_opr.xlsx')
    
    budownictwo=df[df["wyszczegolnienie"].isin(['Wyroby drewniane, papiernictwo i poligrafia',
'Przemysł chemiczny i farmaceutyczny, ceramika i przetwarzanie materiałów niemetalicznych',
'Produkcja metali i metalowych wyrobów gotowych',
'Energia elektryczna, gaz, ciepło',
'Dostawa wody, ścieki i odpady',
'Budownictwo',
'Transport i gospodarka magazynowa',
'Obsługa rynku nieruchomości',
'Administrowanie i działalność wspierająca',
'Wynajem, dzierżawa, doradztwo personalne',
'Ochrona, utrzymanie czystości itp.'])]
    vector_budownictwo = budownictwo["wyszczegolnienie"].tolist()
    df_transformed_bud = budownictwo.set_index("wyszczegolnienie").T
    df_transformed_bud.columns = vector_budownictwo
    df_transformed_bud.index = [2015, 2016, 2017, 2018, 2019, 2020, 2021,2022]
    correlation_matrix = df_transformed_bud.corr()
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
    plt.title("Macierz korelacji")
    plt.show()
    st.pyplot(fig)
    
def budnak_sprzet_corr():
    st.markdown('### Korelacje - przedsiębiorstwa, które poniosły nakłady na sprzęt w branżach związanych z budownictwem')
    
    # plik_wejsciowy =  "C:/Users/Dell/Desktop/streamlit/DATAFRAME/datadzialy_wide_N_sprzet_o.xlsx" 
    # df = pd.read_excel(plik_wejsciowy)
    
    df = pd.read_excel('datadzialy_wide_N_sprzet_o.xlsx')
    
    budownictwo=df[df["wyszczegolnienie"].isin(['Wyroby drewniane, papiernictwo i poligrafia',
'Przemysł chemiczny i farmaceutyczny, ceramika i przetwarzanie materiałów niemetalicznych',
'Produkcja metali i metalowych wyrobów gotowych',
'Energia elektryczna, gaz, ciepło',
'Dostawa wody, ścieki i odpady',
'Budownictwo',
'Transport i gospodarka magazynowa',
'Obsługa rynku nieruchomości',
'Administrowanie i działalność wspierająca',
'Wynajem, dzierżawa, doradztwo personalne',
'Ochrona, utrzymanie czystości itp.'])]
    vector_budownictwo = budownictwo["wyszczegolnienie"].tolist()
    df_transformed_bud = budownictwo.set_index("wyszczegolnienie").T
    df_transformed_bud.columns = vector_budownictwo
    df_transformed_bud.index = [2015, 2016, 2017, 2018, 2019, 2020, 2021,2022]
    correlation_matrix = df_transformed_bud.corr()
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
    plt.title("Macierz korelacji")
    plt.show()
    st.pyplot(fig)

def budnak_sprzet_inf_corr():
    st.markdown('### Korelacje - przedsiębiorstwa, które poniosły nakłady na sprzęt informatyczny w branżach związanych z budownictwem')
    
    # plik_wejsciowy =  "C:/Users/Dell/Desktop/streamlit/DATAFRAME/datadzialy_wide_N_sprzet_inf.xlsx" 
    # df = pd.read_excel(plik_wejsciowy)
    df = pd.read_excel('datadzialy_wide_N_sprzet_inf.xlsx')
    
    budownictwo=df[df["wyszczegolnienie"].isin(['Wyroby drewniane, papiernictwo i poligrafia',
'Przemysł chemiczny i farmaceutyczny, ceramika i przetwarzanie materiałów niemetalicznych',
'Produkcja metali i metalowych wyrobów gotowych',
'Energia elektryczna, gaz, ciepło',
'Dostawa wody, ścieki i odpady',
'Budownictwo',
'Transport i gospodarka magazynowa',
'Obsługa rynku nieruchomości',
'Administrowanie i działalność wspierająca',
'Wynajem, dzierżawa, doradztwo personalne',
'Ochrona, utrzymanie czystości itp.'])]
    vector_budownictwo = budownictwo["wyszczegolnienie"].tolist()
    df_transformed_bud = budownictwo.set_index("wyszczegolnienie").T
    df_transformed_bud.columns = vector_budownictwo
    df_transformed_bud.index = [2015, 2016, 2017, 2018, 2019, 2020, 2021,2022]
    correlation_matrix = df_transformed_bud.corr()
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
    plt.title("Macierz korelacji")
    plt.show()
    st.pyplot(fig)
    
def budnak_sprzet_tele_corr():
    st.markdown('### Korelacje - przedsiębiorstwa, które poniosły nakłady na sprzęt telekomunikacyjny w branżach związanych z budownictwem')
    
    # plik_wejsciowy =  "C:/Users/Dell/Desktop/streamlit/DATAFRAME/datadzialy_wide_N_sprzet_tele.xlsx" 
    # df = pd.read_excel(plik_wejsciowy)
    df = pd.read_excel('datadzialy_wide_N_sprzet_tele.xlsx')
    
    budownictwo=df[df["wyszczegolnienie"].isin(['Wyroby drewniane, papiernictwo i poligrafia',
'Przemysł chemiczny i farmaceutyczny, ceramika i przetwarzanie materiałów niemetalicznych',
'Produkcja metali i metalowych wyrobów gotowych',
'Energia elektryczna, gaz, ciepło',
'Dostawa wody, ścieki i odpady',
'Budownictwo',
'Transport i gospodarka magazynowa',
'Obsługa rynku nieruchomości',
'Administrowanie i działalność wspierająca',
'Wynajem, dzierżawa, doradztwo personalne',
'Ochrona, utrzymanie czystości itp.'])]
    vector_budownictwo = budownictwo["wyszczegolnienie"].tolist()
    df_transformed_bud = budownictwo.set_index("wyszczegolnienie").T
    df_transformed_bud.columns = vector_budownictwo
    df_transformed_bud.index = [2015, 2016, 2017, 2018, 2019, 2020, 2021,2022]
    correlation_matrix = df_transformed_bud.corr()
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
    plt.title("Macierz korelacji")
    plt.show()
    st.pyplot(fig)
    
def budnak_oprog_corr():
    st.markdown('### Korelacje - przedsiębiorstwa, które poniosły nakłady na oprogramowanie w branżach związanych z budownictwem')
    
    # plik_wejsciowy =  "C:/Users/Dell/Desktop/streamlit/DATAFRAME/datadzialy_wide_N_oprog.xlsx" 
    # df = pd.read_excel(plik_wejsciowy)
    df = pd.read_excel('datadzialy_wide_N_oprog.xlsx')
    
    budownictwo=df[df["wyszczegolnienie"].isin(['Wyroby drewniane, papiernictwo i poligrafia',
'Przemysł chemiczny i farmaceutyczny, ceramika i przetwarzanie materiałów niemetalicznych',
'Produkcja metali i metalowych wyrobów gotowych',
'Energia elektryczna, gaz, ciepło',
'Dostawa wody, ścieki i odpady',
'Budownictwo',
'Transport i gospodarka magazynowa',
'Obsługa rynku nieruchomości',
'Administrowanie i działalność wspierająca',
'Wynajem, dzierżawa, doradztwo personalne',
'Ochrona, utrzymanie czystości itp.'])]
    vector_budownictwo = budownictwo["wyszczegolnienie"].tolist()
    df_transformed_bud = budownictwo.set_index("wyszczegolnienie").T
    df_transformed_bud.columns = vector_budownictwo
    df_transformed_bud.index = [2015, 2016, 2017, 2018, 2019, 2020, 2021,2022]
    correlation_matrix = df_transformed_bud.corr()
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
    plt.title("Macierz korelacji")
    plt.show()
    st.pyplot(fig)
    
def budcorr_wartosc_nakladow_section():
    st.sidebar.header("Wartość Nakładów")
    pages = {
        "Wartość nakładów - sprzęt/oprogramowanie": budwnak_sprzet_opr_corr,
        "Wartość nakładów - sprzęt": budwnak_sprzet_corr,
        "Wartość nakładów - sprzęt informatyczny": budwnak_sprzet_inf_corr,
        "Wartość nakładów - sprzęt teleinformatyczny": budwnak_sprzet_tele_corr,
        "Wartość nakładów - leasing": budwnak_leasing_corr,
        "Wartość nakładów - oprogramowanie": budwnak_oprog_corr,
    }
    selected_page = st.sidebar.selectbox("Wybierz analizę", pages.keys())
    pages[selected_page]()
    
def budwnak_leasing_corr():
    st.markdown('### Korelacje - wartość nakładów poniesionych na leasing urządzeń ICT w przedsiębiorstwach związanych z budownictwem')
    
    # plik_wejsciowy =  "C:/Users/Dell/Desktop/streamlit/DATAFRAME/datadzialy_wide_WN_leasing.xlsx"
    # df = pd.read_excel(plik_wejsciowy)
    df = pd.read_excel('datadzialy_wide_WN_leasing.xlsx')
    budownictwo=df[df["wyszczegolnienie"].isin(['Wyroby drewniane, papiernictwo i poligrafia',
'Przemysł chemiczny i farmaceutyczny, ceramika i przetwarzanie materiałów niemetalicznych',
'Produkcja metali i metalowych wyrobów gotowych',
'Energia elektryczna, gaz, ciepło',
'Dostawa wody, ścieki i odpady',
'Budownictwo',
'Transport i gospodarka magazynowa',
'Obsługa rynku nieruchomości',
'Administrowanie i działalność wspierająca',
'Wynajem, dzierżawa, doradztwo personalne',
'Ochrona, utrzymanie czystości itp.'])]
    vector_budownictwo = budownictwo["wyszczegolnienie"].tolist()
    df_transformed_bud = budownictwo.set_index("wyszczegolnienie").T
    df_transformed_bud.columns = vector_budownictwo
    df_transformed_bud.index = [2015, 2016, 2017, 2018, 2019, 2020, 2021,2022]
    correlation_matrix = df_transformed_bud.corr()
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
    plt.title("Macierz korelacji")
    plt.show()
    st.pyplot(fig)
    
def budwnak_sprzet_opr_corr():
    st.markdown('### Korelacje - wartość nakładów poniesionych na sprzęt i oprogramowanie w przedsiębiorstwach związanych z budownictwem')
    
    # plik_wejsciowy =  "C:/Users/Dell/Desktop/streamlit/DATAFRAME/datadzialy_wide_WN_sprzet_opr.xlsx" 
    # df = pd.read_excel(plik_wejsciowy)
    df = pd.read_excel('datadzialy_wide_WN_sprzet_opr.xlsx')
    budownictwo=df[df["wyszczegolnienie"].isin(['Wyroby drewniane, papiernictwo i poligrafia',
'Przemysł chemiczny i farmaceutyczny, ceramika i przetwarzanie materiałów niemetalicznych',
'Produkcja metali i metalowych wyrobów gotowych',
'Energia elektryczna, gaz, ciepło',
'Dostawa wody, ścieki i odpady',
'Budownictwo',
'Transport i gospodarka magazynowa',
'Obsługa rynku nieruchomości',
'Administrowanie i działalność wspierająca',
'Wynajem, dzierżawa, doradztwo personalne',
'Ochrona, utrzymanie czystości itp.'])]
    vector_budownictwo = budownictwo["wyszczegolnienie"].tolist()
    df_transformed_bud = budownictwo.set_index("wyszczegolnienie").T
    df_transformed_bud.columns = vector_budownictwo
    df_transformed_bud.index = [2015, 2016, 2017, 2018, 2019, 2020, 2021,2022]
    correlation_matrix = df_transformed_bud.corr()
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
    plt.title("Macierz korelacji")
    plt.show()
    st.pyplot(fig)
    
def budwnak_sprzet_corr():
    st.markdown('### Korelacje - wartość nakładów poniesionych na sprzęt w przedsiębiorstwach związanych z budownictwem')
    
    # plik_wejsciowy =  "C:/Users/Dell/Desktop/streamlit/DATAFRAME/datadzialy_wide_WN_sprzet_o.xlsx" 
    # df = pd.read_excel(plik_wejsciowy)
    df = pd.read_excel('datadzialy_wide_WN_sprzet_o.xlsx')
    
    budownictwo=df[df["wyszczegolnienie"].isin(['Wyroby drewniane, papiernictwo i poligrafia',
'Przemysł chemiczny i farmaceutyczny, ceramika i przetwarzanie materiałów niemetalicznych',
'Produkcja metali i metalowych wyrobów gotowych',
'Energia elektryczna, gaz, ciepło',
'Dostawa wody, ścieki i odpady',
'Budownictwo',
'Transport i gospodarka magazynowa',
'Obsługa rynku nieruchomości',
'Administrowanie i działalność wspierająca',
'Wynajem, dzierżawa, doradztwo personalne',
'Ochrona, utrzymanie czystości itp.'])]
    vector_budownictwo = budownictwo["wyszczegolnienie"].tolist()
    df_transformed_bud = budownictwo.set_index("wyszczegolnienie").T
    df_transformed_bud.columns = vector_budownictwo
    df_transformed_bud.index = [2015, 2016, 2017, 2018, 2019, 2020, 2021,2022]
    correlation_matrix = df_transformed_bud.corr()
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
    plt.title("Macierz korelacji")
    plt.show()
    st.pyplot(fig)

def budwnak_sprzet_inf_corr():
    st.markdown('### Korelacje - wartość nakładów poniesionych na sprzęt informatyczny w przedsiębiorstwach związanych z budownictwem')
    
    # plik_wejsciowy =  "C:/Users/Dell/Desktop/streamlit/DATAFRAME/datadzialy_wide_WN_sprzet_inf.xlsx" 
    # df = pd.read_excel(plik_wejsciowy)
    df = pd.read_excel('datadzialy_wide_WN_sprzet_inf.xlsx')
    budownictwo=df[df["wyszczegolnienie"].isin(['Wyroby drewniane, papiernictwo i poligrafia',
'Przemysł chemiczny i farmaceutyczny, ceramika i przetwarzanie materiałów niemetalicznych',
'Produkcja metali i metalowych wyrobów gotowych',
'Energia elektryczna, gaz, ciepło',
'Dostawa wody, ścieki i odpady',
'Budownictwo',
'Transport i gospodarka magazynowa',
'Obsługa rynku nieruchomości',
'Administrowanie i działalność wspierająca',
'Wynajem, dzierżawa, doradztwo personalne',
'Ochrona, utrzymanie czystości itp.'])]
    vector_budownictwo = budownictwo["wyszczegolnienie"].tolist()
    df_transformed_bud = budownictwo.set_index("wyszczegolnienie").T
    df_transformed_bud.columns = vector_budownictwo
    df_transformed_bud.index = [2015, 2016, 2017, 2018, 2019, 2020, 2021,2022]
    correlation_matrix = df_transformed_bud.corr()
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
    plt.title("Macierz korelacji")
    plt.show()
    st.pyplot(fig)
    
def budwnak_sprzet_tele_corr():
    st.markdown('### Korelacje - wartość nakładów poniesionych na sprzęt telekomunikacyjny w przedsiębiorstwach związanych z budownictwem')
    
    # plik_wejsciowy =  "C:/Users/Dell/Desktop/streamlit/DATAFRAME/datadzialy_wide_WN_sprzet_tele.xlsx" 
    # df = pd.read_excel(plik_wejsciowy)
    df = pd.read_excel('datadzialy_wide_WN_sprzet_tele.xlsx')
    budownictwo=df[df["wyszczegolnienie"].isin(['Wyroby drewniane, papiernictwo i poligrafia',
'Przemysł chemiczny i farmaceutyczny, ceramika i przetwarzanie materiałów niemetalicznych',
'Produkcja metali i metalowych wyrobów gotowych',
'Energia elektryczna, gaz, ciepło',
'Dostawa wody, ścieki i odpady',
'Budownictwo',
'Transport i gospodarka magazynowa',
'Obsługa rynku nieruchomości',
'Administrowanie i działalność wspierająca',
'Wynajem, dzierżawa, doradztwo personalne',
'Ochrona, utrzymanie czystości itp.'])]
    vector_budownictwo = budownictwo["wyszczegolnienie"].tolist()
    df_transformed_bud = budownictwo.set_index("wyszczegolnienie").T
    df_transformed_bud.columns = vector_budownictwo
    df_transformed_bud.index = [2015, 2016, 2017, 2018, 2019, 2020, 2021,2022]
    correlation_matrix = df_transformed_bud.corr()
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
    plt.title("Macierz korelacji")
    plt.show()
    st.pyplot(fig)
    
def budwnak_oprog_corr():
    st.markdown('### Korelacje - wartość nakładów poniesionych na oprogramowanie w przedsiębiorstwach związanych z budownictwem')
    
    # plik_wejsciowy =  "C:/Users/Dell/Desktop/streamlit/DATAFRAME/datadzialy_wide_WN_oprog.xlsx"
    # df = pd.read_excel(plik_wejsciowy)
    df = pd.read_excel('datadzialy_wide_WN_oprog.xlsx')
    budownictwo=df[df["wyszczegolnienie"].isin(['Wyroby drewniane, papiernictwo i poligrafia',
'Przemysł chemiczny i farmaceutyczny, ceramika i przetwarzanie materiałów niemetalicznych',
'Produkcja metali i metalowych wyrobów gotowych',
'Energia elektryczna, gaz, ciepło',
'Dostawa wody, ścieki i odpady',
'Budownictwo',
'Transport i gospodarka magazynowa',
'Obsługa rynku nieruchomości',
'Administrowanie i działalność wspierająca',
'Wynajem, dzierżawa, doradztwo personalne',
'Ochrona, utrzymanie czystości itp.'])]
    vector_budownictwo = budownictwo["wyszczegolnienie"].tolist()
    df_transformed_bud = budownictwo.set_index("wyszczegolnienie").T
    df_transformed_bud.columns = vector_budownictwo
    df_transformed_bud.index = [2015, 2016, 2017, 2018, 2019, 2020, 2021,2022]
    correlation_matrix = df_transformed_bud.corr()
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
    plt.title("Macierz korelacji")
    plt.show()
    st.pyplot(fig)    
    
def budcorr_pracownicy_section():
    st.sidebar.header("Pracownicy")
    pages = {
        "Przedsiębiorstwa zatrudniające pracowników": budprzed_o_corr,
        "Przedsiębiorstwa z dostępem do internetu": budprzed_di_corr,
        "Pracujący ogółem": budpr_o_corr,
        "Pracownicy z wyższym wykształceniem": budpr_ww_corr,
        "Pracownicy z dostępem do internetu": budpr_di_corr,
        "Pracownicy z urządzeniami przenośnymi": budpr_up_corr,
    }
    selected_page = st.sidebar.selectbox("Wybierz analizę", pages.keys())
    pages[selected_page]()
    
def budprzed_o_corr():
    st.markdown('### Korelacje - ilość przedsiębiorstw w branżach związanych z budownictwem')
    
    # plik_wejsciowy =  "C:/Users/Dell/Desktop/streamlit/DATAFRAME/datadzialy_wide_P_przed_o.xlsx"
    # df = pd.read_excel(plik_wejsciowy)
    df = pd.read_excel('datadzialy_wide_P_przed_o.xlsx')
    budownictwo=df[df["wyszczegolnienie"].isin(['Wyroby drewniane, papiernictwo i poligrafia',
'Przemysł chemiczny i farmaceutyczny, ceramika i przetwarzanie materiałów niemetalicznych',
'Produkcja metali i metalowych wyrobów gotowych',
'Energia elektryczna, gaz, ciepło',
'Dostawa wody, ścieki i odpady',
'Budownictwo',
'Transport i gospodarka magazynowa',
'Obsługa rynku nieruchomości',
'Administrowanie i działalność wspierająca',
'Wynajem, dzierżawa, doradztwo personalne',
'Ochrona, utrzymanie czystości itp.'])]
    vector_budownictwo = budownictwo["wyszczegolnienie"].tolist()
    df_transformed_bud = budownictwo.set_index("wyszczegolnienie").T
    df_transformed_bud.columns = vector_budownictwo
    df_transformed_bud.index = [2015, 2016, 2017, 2018, 2019, 2020, 2021]
    correlation_matrix = df_transformed_bud.corr()
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
    plt.title("Macierz korelacji")
    plt.show()
    st.pyplot(fig)
    
def budprzed_di_corr():
    st.markdown('### Korelacje - ilość przedsiębiorstw z dostępem do internetu w branżach związanych z budownictwem')
    
    # plik_wejsciowy =  "C:/Users/Dell/Desktop/streamlit/DATAFRAME/datadzialy_wide_P_przed_di.xlsx" 
    # df = pd.read_excel(plik_wejsciowy)
    df = pd.read_excel('datadzialy_wide_P_przed_di.xlsx')
    budownictwo=df[df["wyszczegolnienie"].isin(['Wyroby drewniane, papiernictwo i poligrafia',
'Przemysł chemiczny i farmaceutyczny, ceramika i przetwarzanie materiałów niemetalicznych',
'Produkcja metali i metalowych wyrobów gotowych',
'Energia elektryczna, gaz, ciepło',
'Dostawa wody, ścieki i odpady',
'Budownictwo',
'Transport i gospodarka magazynowa',
'Obsługa rynku nieruchomości',
'Administrowanie i działalność wspierająca',
'Wynajem, dzierżawa, doradztwo personalne',
'Ochrona, utrzymanie czystości itp.'])]
    vector_budownictwo = budownictwo["wyszczegolnienie"].tolist()
    df_transformed_bud = budownictwo.set_index("wyszczegolnienie").T
    df_transformed_bud.columns = vector_budownictwo
    df_transformed_bud.index = [2015, 2016, 2017, 2018, 2019, 2020, 2021]
    correlation_matrix = df_transformed_bud.corr()
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
    plt.title("Macierz korelacji")
    plt.show()
    st.pyplot(fig)
    
def budpr_o_corr():
    st.markdown('### Korelacje - ilość pracujących ogółem w branżach związanych z budownictwem')
    
    # plik_wejsciowy =  "C:/Users/Dell/Desktop/streamlit/DATAFRAME/datadzialy_wide_P_pr_o.xlsx"
    # df = pd.read_excel(plik_wejsciowy)
    df = pd.read_excel('datadzialy_wide_P_pr_o.xlsx')
    budownictwo=df[df["wyszczegolnienie"].isin(['Wyroby drewniane, papiernictwo i poligrafia',
'Przemysł chemiczny i farmaceutyczny, ceramika i przetwarzanie materiałów niemetalicznych',
'Produkcja metali i metalowych wyrobów gotowych',
'Energia elektryczna, gaz, ciepło',
'Dostawa wody, ścieki i odpady',
'Budownictwo',
'Transport i gospodarka magazynowa',
'Obsługa rynku nieruchomości',
'Administrowanie i działalność wspierająca',
'Wynajem, dzierżawa, doradztwo personalne',
'Ochrona, utrzymanie czystości itp.'])]
    vector_budownictwo = budownictwo["wyszczegolnienie"].tolist()
    df_transformed_bud = budownictwo.set_index("wyszczegolnienie").T
    df_transformed_bud.columns = vector_budownictwo
    df_transformed_bud.index = [2015, 2016, 2017, 2018, 2019, 2020, 2021]
    correlation_matrix = df_transformed_bud.corr()
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
    plt.title("Macierz korelacji")
    plt.show()
    st.pyplot(fig)
    
def budpr_ww_corr():
    st.markdown('### Korelacje - ilość pracowników z wyższym wykształceniem w branżach związanych z budownictwem')
    
    # plik_wejsciowy =  "C:/Users/Dell/Desktop/streamlit/DATAFRAME/datadzialy_wide_P_pr_ww.xlsx"
    # df = pd.read_excel(plik_wejsciowy)
    df = pd.read_excel('datadzialy_wide_P_pr_ww.xlsx')
    budownictwo=df[df["wyszczegolnienie"].isin(['Wyroby drewniane, papiernictwo i poligrafia',
'Przemysł chemiczny i farmaceutyczny, ceramika i przetwarzanie materiałów niemetalicznych',
'Produkcja metali i metalowych wyrobów gotowych',
'Energia elektryczna, gaz, ciepło',
'Dostawa wody, ścieki i odpady',
'Budownictwo',
'Transport i gospodarka magazynowa',
'Obsługa rynku nieruchomości',
'Administrowanie i działalność wspierająca',
'Wynajem, dzierżawa, doradztwo personalne',
'Ochrona, utrzymanie czystości itp.'])]
    vector_budownictwo = budownictwo["wyszczegolnienie"].tolist()
    df_transformed_bud = budownictwo.set_index("wyszczegolnienie").T
    df_transformed_bud.columns = vector_budownictwo
    df_transformed_bud.index = [2015, 2016, 2017, 2018, 2019, 2020, 2021]
    correlation_matrix = df_transformed_bud.corr()
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
    plt.title("Macierz korelacji")
    plt.show()
    st.pyplot(fig)
    
def budpr_di_corr():
    st.markdown('### Korelacje - ilość pracowników z dostępem do internetu w branżach związanych z budownictwem')
    
    # plik_wejsciowy =  "C:/Users/Dell/Desktop/streamlit/DATAFRAME/datadzialy_wide_P_pr_di.xlsx"  
    # df = pd.read_excel(plik_wejsciowy)
    df = pd.read_excel('datadzialy_wide_P_pr_di.xlsx')
    
    budownictwo=df[df["wyszczegolnienie"].isin(['Wyroby drewniane, papiernictwo i poligrafia',
'Przemysł chemiczny i farmaceutyczny, ceramika i przetwarzanie materiałów niemetalicznych',
'Produkcja metali i metalowych wyrobów gotowych',
'Energia elektryczna, gaz, ciepło',
'Dostawa wody, ścieki i odpady',
'Budownictwo',
'Transport i gospodarka magazynowa',
'Obsługa rynku nieruchomości',
'Administrowanie i działalność wspierająca',
'Wynajem, dzierżawa, doradztwo personalne',
'Ochrona, utrzymanie czystości itp.'])]
    vector_budownictwo = budownictwo["wyszczegolnienie"].tolist()
    df_transformed_bud = budownictwo.set_index("wyszczegolnienie").T
    df_transformed_bud.columns = vector_budownictwo
    df_transformed_bud.index = [2015, 2016, 2017, 2018, 2019, 2020, 2021]
    correlation_matrix = df_transformed_bud.corr()
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
    plt.title("Macierz korelacji")
    plt.show()
    st.pyplot(fig)
    
def budpr_up_corr():
    st.markdown('### Korelacje - ilość pracowników dysponujących urządzeniami przenośnymi w branżach związanych z budownictwem')
    
    # plik_wejsciowy =  "C:/Users/Dell/Desktop/streamlit/DATAFRAME/datadzialy_wide_P_pr_up.xlsx" 
    # df = pd.read_excel(plik_wejsciowy)
    df = pd.read_excel('datadzialy_wide_P_pr_up.xlsx')
    
    budownictwo=df[df["wyszczegolnienie"].isin(['Wyroby drewniane, papiernictwo i poligrafia',
'Przemysł chemiczny i farmaceutyczny, ceramika i przetwarzanie materiałów niemetalicznych',
'Produkcja metali i metalowych wyrobów gotowych',
'Energia elektryczna, gaz, ciepło',
'Dostawa wody, ścieki i odpady',
'Budownictwo',
'Transport i gospodarka magazynowa',
'Obsługa rynku nieruchomości',
'Administrowanie i działalność wspierająca',
'Wynajem, dzierżawa, doradztwo personalne',
'Ochrona, utrzymanie czystości itp.'])]
    vector_budownictwo = budownictwo["wyszczegolnienie"].tolist()
    df_transformed_bud = budownictwo.set_index("wyszczegolnienie").T
    df_transformed_bud.columns = vector_budownictwo
    df_transformed_bud.index = [2015, 2016, 2017, 2018, 2019, 2020, 2021]
    correlation_matrix = df_transformed_bud.corr()
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
    plt.title("Macierz korelacji")
    plt.show()
    st.pyplot(fig)
    
def budcorr_rodzaje_polaczen_section():
    st.sidebar.header("Rodzaje Połączeń")
    pages = {
        "szerokopasmowy  dostęp do Internetu": budrodz_pol_szer_corr,
        "szerokopasmowy  dostęp do Internetu poprzez łącze DSL": budrodz_pol_szerDSL_corr,
    }
    selected_page = st.sidebar.selectbox("Wybierz analizę", pages.keys())
    pages[selected_page]()
    
def budrodz_pol_szer_corr():
    st.markdown('### Korelacje - przedsiębiorstwa z dostępem do internetu poprzez łącze szerokopasmowe w branżach związanych z budownictwem')
    
    # plik_wejsciowy =  "C:/Users/Dell/Desktop/streamlit/DATAFRAME/datadzialy_wide_RP_od_szer_o.xlsx" 
    # df = pd.read_excel(plik_wejsciowy)
    df = pd.read_excel('datadzialy_wide_RP_od_szer_o.xlsx')
    
    budownictwo=df[df["wyszczegolnienie"].isin(['Wyroby drewniane, papiernictwo i poligrafia',
'Przemysł chemiczny i farmaceutyczny, ceramika i przetwarzanie materiałów niemetalicznych',
'Produkcja metali i metalowych wyrobów gotowych',
'Energia elektryczna, gaz, ciepło',
'Dostawa wody, ścieki i odpady',
'Budownictwo',
'Transport i gospodarka magazynowa',
'Obsługa rynku nieruchomości',
'Administrowanie i działalność wspierająca',
'Wynajem, dzierżawa, doradztwo personalne',
'Ochrona, utrzymanie czystości itp.'])]
    vector_budownictwo = budownictwo["wyszczegolnienie"].tolist()
    df_transformed_bud = budownictwo.set_index("wyszczegolnienie").T
    df_transformed_bud.columns = vector_budownictwo
    df_transformed_bud.index = [2015, 2016, 2017, 2018, 2019, 2020, 2021,2022]
    correlation_matrix = df_transformed_bud.corr()
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
    plt.title("Macierz korelacji")
    plt.show()
    st.pyplot(fig)
    
def budrodz_pol_szerDSL_corr():
    st.markdown('### Korelacje - przedsiębiorstwa z dostępem do internetu poprzez łącze szerokopasmowe wykorzystywane z użyciem technologii DSL w branżach związanych z budownictwem')
    
    # plik_wejsciowy =  "C:/Users/Dell/Desktop/streamlit/DATAFRAME/datadzialy_wide_RP_od_szer_DSL.xlsx"
    # df = pd.read_excel(plik_wejsciowy)
    df = pd.read_excel('datadzialy_wide_RP_od_szer_DSL.xlsx')
    budownictwo=df[df["wyszczegolnienie"].isin(['Wyroby drewniane, papiernictwo i poligrafia',
'Przemysł chemiczny i farmaceutyczny, ceramika i przetwarzanie materiałów niemetalicznych',
'Produkcja metali i metalowych wyrobów gotowych',
'Energia elektryczna, gaz, ciepło',
'Dostawa wody, ścieki i odpady',
'Budownictwo',
'Transport i gospodarka magazynowa',
'Obsługa rynku nieruchomości',
'Administrowanie i działalność wspierająca',
'Wynajem, dzierżawa, doradztwo personalne',
'Ochrona, utrzymanie czystości itp.'])]
    vector_budownictwo = budownictwo["wyszczegolnienie"].tolist()
    df_transformed_bud = budownictwo.set_index("wyszczegolnienie").T
    df_transformed_bud.columns = vector_budownictwo
    df_transformed_bud.index = [2015, 2016, 2017, 2018, 2019, 2020, 2021,2022]
    correlation_matrix = df_transformed_bud.corr()
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
    plt.title("Macierz korelacji")
    plt.show()
    st.pyplot(fig)
    
def budcorr_predkosc_polaczen_section():
    st.sidebar.header("Prędkość Połączeń")
    pages = {
        "szerokopasmowy  dostęp do Internetu - prędkość połączenia mniej niż 100 Mbit/s": budpred_pol_szer_w_corr,
        "szerokopasmowy  dostęp do Internetu - prędkość połączenia przynajmniej 100 Mbit/s": budpred_pol_szer_s_corr
    }
    selected_page = st.sidebar.selectbox("Wybierz analizę", pages.keys())
    pages[selected_page]()
    
def budpred_pol_szer_w_corr():
    st.markdown('### Korelacje - prędkość połączenia mniej niż 100 Mbit/s w branżach związanych z budownictwem')
    
    # plik_wejsciowy =  "C:/Users/Dell/Desktop/streamlit/DATAFRAME/datadzialy_wide_PP_od_szer_w.xlsx"
    # df = pd.read_excel(plik_wejsciowy)
    df = pd.read_excel('datadzialy_wide_PP_od_szer_w.xlsx')
    budownictwo=df[df["wyszczegolnienie"].isin(['Wyroby drewniane, papiernictwo i poligrafia',
'Przemysł chemiczny i farmaceutyczny, ceramika i przetwarzanie materiałów niemetalicznych',
'Produkcja metali i metalowych wyrobów gotowych',
'Energia elektryczna, gaz, ciepło',
'Dostawa wody, ścieki i odpady',
'Budownictwo',
'Transport i gospodarka magazynowa',
'Obsługa rynku nieruchomości',
'Administrowanie i działalność wspierająca',
'Wynajem, dzierżawa, doradztwo personalne',
'Ochrona, utrzymanie czystości itp.'])]
    vector_budownictwo = budownictwo["wyszczegolnienie"].tolist()
    df_transformed_bud = budownictwo.set_index("wyszczegolnienie").T
    df_transformed_bud.columns = vector_budownictwo
    df_transformed_bud.index = [2015, 2016, 2017, 2018, 2019, 2020, 2021,2022]
    correlation_matrix = df_transformed_bud.corr()
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
    plt.title("Macierz korelacji")
    plt.show()
    st.pyplot(fig)
    
def budpred_pol_szer_s_corr():
    st.markdown('### Korelacje - prędkość połączenia przynajmniej 100 Mbit/s w branżach związanych z budownictwem')
    
    # plik_wejsciowy =  "C:/Users/Dell/Desktop/streamlit/DATAFRAME/datadzialy_wide_PP_od_szer_s.xlsx" 
    # df = pd.read_excel(plik_wejsciowy)
    df = pd.read_excel('datadzialy_wide_PP_od_szer_s.xlsx')
    
    budownictwo=df[df["wyszczegolnienie"].isin(['Wyroby drewniane, papiernictwo i poligrafia',
'Przemysł chemiczny i farmaceutyczny, ceramika i przetwarzanie materiałów niemetalicznych',
'Produkcja metali i metalowych wyrobów gotowych',
'Energia elektryczna, gaz, ciepło',
'Dostawa wody, ścieki i odpady',
'Budownictwo',
'Transport i gospodarka magazynowa',
'Obsługa rynku nieruchomości',
'Administrowanie i działalność wspierająca',
'Wynajem, dzierżawa, doradztwo personalne',
'Ochrona, utrzymanie czystości itp.'])]
    vector_budownictwo = budownictwo["wyszczegolnienie"].tolist()
    df_transformed_bud = budownictwo.set_index("wyszczegolnienie").T
    df_transformed_bud.columns = vector_budownictwo
    df_transformed_bud.index = [2015, 2016, 2017, 2018, 2019, 2020, 2021,2022]
    correlation_matrix = df_transformed_bud.corr()
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
    plt.title("Macierz korelacji")
    plt.show()
    st.pyplot(fig)

############################KORELACJE - TURYSTYKA#######################################################################################
def korelacje_tur_section():
    selected_subsection = st.sidebar.selectbox("Wybierz kategorię", [
        "Nakłady", "Pracownicy", "Wartość Nakładów", "Rodzaje Połączeń", "Prędkość Połączeń"
    ])
    if selected_subsection == "Nakłady":
        turcorr_naklady_section()
    elif selected_subsection == "Pracownicy":
        turcorr_pracownicy_section()
    elif selected_subsection == "Wartość Nakładów":
        turcorr_wartosc_nakladow_section()
    elif selected_subsection == "Rodzaje Połączeń":
        turcorr_rodzaje_polaczen_section()
    elif selected_subsection == "Prędkość Połączeń":
        turcorr_predkosc_polaczen_section()
        
def turcorr_naklady_section():
    st.sidebar.header("Nakłady")
    pages = {
        "Nakłady na sprzęt/oprogramowanie": turnak_sprzet_opr_corr,
        "Nakłady - sprzęt ogółem": turnak_sprzet_corr,
        "Nakłady - sprzęt informatyczny": turnak_sprzet_inf_corr,
        "Nakłady - sprzęt telekomunikacyjny": turnak_sprzet_tele_corr,
        "Nakłady - leasing": turnak_leasing_corr,
        "Nakłady - oprogramowanie": turnak_oprog_corr,
    }
    selected_page = st.sidebar.selectbox("Wybierz analizę", pages.keys())
    pages[selected_page]()
    
def turnak_leasing_corr():
    st.markdown('### Korelacje - przedsiębiorstwa, które poniosły nakłady na leasing urządzeń ICT w branżach związanych z turystyką')
    
    # plik_wejsciowy =  "C:/Users/Dell/Desktop/streamlit/DATAFRAME/datadzialy_wide_N_leasing.xlsx" 
    # df = pd.read_excel(plik_wejsciowy)
    df = pd.read_excel('datadzialy_wide_N_leasing.xlsx')
    
    turystyka=df[df["wyszczegolnienie"].isin(['Zakwaterowanie','Wyżywienie','Turystyka'])]
    vector_turystyka = turystyka["wyszczegolnienie"].tolist()
    df_transformed_tur = turystyka.set_index("wyszczegolnienie").T
    df_transformed_tur.columns = vector_turystyka
    df_transformed_tur.index = [2015, 2016, 2017, 2018, 2019, 2020, 2021,2022]
    correlation_matrix = df_transformed_tur.corr()
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
    plt.title("Macierz korelacji")
    plt.show()
    st.pyplot(fig)
    
def turnak_sprzet_opr_corr():
    st.markdown('### Korelacje - przedsiębiorstwa, które poniosły nakłady na sprzęt i oprogramowanie w branżach związanych z turystyką')
    
    # plik_wejsciowy =  "C:/Users/Dell/Desktop/streamlit/DATAFRAME/datadzialy_wide_N_sprzet_opr.xlsx" 
    # df = pd.read_excel(plik_wejsciowy)
    df = pd.read_excel('datadzialy_wide_N_sprzet_opr.xlsx')
    
    turystyka=df[df["wyszczegolnienie"].isin(['Zakwaterowanie','Wyżywienie','Turystyka'])]
    vector_turystyka = turystyka["wyszczegolnienie"].tolist()
    df_transformed_tur = turystyka.set_index("wyszczegolnienie").T
    df_transformed_tur.columns = vector_turystyka
    df_transformed_tur.index = [2015, 2016, 2017, 2018, 2019, 2020, 2021,2022]
    correlation_matrix = df_transformed_tur.corr()
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
    plt.title("Macierz korelacji")
    plt.show()
    st.pyplot(fig)
    
def turnak_sprzet_corr():
    st.markdown('### Korelacje - przedsiębiorstwa, które poniosły nakłady na sprzęt w branżach związanych z turystyką')
    
    # plik_wejsciowy =  "C:/Users/Dell/Desktop/streamlit/DATAFRAME/datadzialy_wide_N_sprzet_o.xlsx"  
    # df = pd.read_excel(plik_wejsciowy)
    df = pd.read_excel('datadzialy_wide_N_sprzet_o.xlsx')
    turystyka=df[df["wyszczegolnienie"].isin(['Zakwaterowanie','Wyżywienie','Turystyka'])]
    vector_turystyka = turystyka["wyszczegolnienie"].tolist()
    df_transformed_tur = turystyka.set_index("wyszczegolnienie").T
    df_transformed_tur.columns = vector_turystyka
    df_transformed_tur.index = [2015, 2016, 2017, 2018, 2019, 2020, 2021,2022]
    correlation_matrix = df_transformed_tur.corr()
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
    plt.title("Macierz korelacji")
    plt.show()
    st.pyplot(fig)

def turnak_sprzet_inf_corr():
    st.markdown('### Korelacje - przedsiębiorstwa, które poniosły nakłady na sprzęt informatyczny w branżach związanych z turystyką')
    
    # plik_wejsciowy =  "C:/Users/Dell/Desktop/streamlit/DATAFRAME/datadzialy_wide_N_sprzet_inf.xlsx"
    # df = pd.read_excel(plik_wejsciowy)
    df = pd.read_excel('datadzialy_wide_N_sprzet_inf.xlsx')
    
    turystyka=df[df["wyszczegolnienie"].isin(['Zakwaterowanie','Wyżywienie','Turystyka'])]
    vector_turystyka = turystyka["wyszczegolnienie"].tolist()
    df_transformed_tur = turystyka.set_index("wyszczegolnienie").T
    df_transformed_tur.columns = vector_turystyka
    df_transformed_tur.index = [2015, 2016, 2017, 2018, 2019, 2020, 2021,2022]
    correlation_matrix = df_transformed_tur.corr()
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
    plt.title("Macierz korelacji")
    plt.show()
    st.pyplot(fig)
    
def turnak_sprzet_tele_corr():
    st.markdown('### Korelacje - przedsiębiorstwa, które poniosły nakłady na sprzęt telekomunikacyjny w branżach związanych z turystyką')
    
    # plik_wejsciowy =  "C:/Users/Dell/Desktop/streamlit/DATAFRAME/datadzialy_wide_N_sprzet_tele.xlsx" 
    # df = pd.read_excel(plik_wejsciowy)
    df = pd.read_excel('datadzialy_wide_N_sprzet_tele.xlsx')
    turystyka=df[df["wyszczegolnienie"].isin(['Zakwaterowanie','Wyżywienie','Turystyka'])]
    vector_turystyka = turystyka["wyszczegolnienie"].tolist()
    df_transformed_tur = turystyka.set_index("wyszczegolnienie").T
    df_transformed_tur.columns = vector_turystyka
    df_transformed_tur.index = [2015, 2016, 2017, 2018, 2019, 2020, 2021,2022]
    correlation_matrix = df_transformed_tur.corr()
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
    plt.title("Macierz korelacji")
    plt.show()
    st.pyplot(fig)
    
def turnak_oprog_corr():
    st.markdown('### Korelacje - przedsiębiorstwa, które poniosły nakłady na oprogramowanie w branżach związanych z turystyką')
    
    # plik_wejsciowy =  "C:/Users/Dell/Desktop/streamlit/DATAFRAME/datadzialy_wide_N_oprog.xlsx" 
    # df = pd.read_excel(plik_wejsciowy)
    df = pd.read_excel('datadzialy_wide_N_oprog.xlsx')
    
    turystyka=df[df["wyszczegolnienie"].isin(['Zakwaterowanie','Wyżywienie','Turystyka'])]
    vector_turystyka = turystyka["wyszczegolnienie"].tolist()
    df_transformed_tur = turystyka.set_index("wyszczegolnienie").T
    df_transformed_tur.columns = vector_turystyka
    df_transformed_tur.index = [2015, 2016, 2017, 2018, 2019, 2020, 2021,2022]
    correlation_matrix = df_transformed_tur.corr()
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
    plt.title("Macierz korelacji")
    plt.show()
    st.pyplot(fig)
    
def turcorr_wartosc_nakladow_section():
    st.sidebar.header("Wartość Nakładów")
    pages = {
        "Wartość nakładów - sprzęt/oprogramowanie": turwnak_sprzet_opr_corr,
        "Wartość nakładów - sprzęt": turwnak_sprzet_corr,
        "Wartość nakładów - sprzęt informatyczny": turwnak_sprzet_inf_corr,
        "Wartość nakładów - sprzęt teleinformatyczny": turwnak_sprzet_tele_corr,
        "Wartość nakładów - leasing": turwnak_leasing_corr,
        "Wartość nakładów - oprogramowanie": turwnak_oprog_corr,
    }
    selected_page = st.sidebar.selectbox("Wybierz analizę", pages.keys())
    pages[selected_page]()
    
def turwnak_leasing_corr():
    st.markdown('### Korelacje - wartość nakładów poniesionych na leasing urządzeń ICT w branżach związanych z turystyką')
    
    # plik_wejsciowy =  "C:/Users/Dell/Desktop/streamlit/DATAFRAME/datadzialy_wide_WN_leasing.xlsx"  
    # df = pd.read_excel(plik_wejsciowy)
    df = pd.read_excel('datadzialy_wide_WN_leasing.xlsx')
    
    turystyka=df[df["wyszczegolnienie"].isin(['Zakwaterowanie','Wyżywienie','Turystyka'])]
    vector_turystyka = turystyka["wyszczegolnienie"].tolist()
    df_transformed_tur = turystyka.set_index("wyszczegolnienie").T
    df_transformed_tur.columns = vector_turystyka
    df_transformed_tur.index = [2015, 2016, 2017, 2018, 2019, 2020, 2021,2022]
    correlation_matrix = df_transformed_tur.corr()
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
    plt.title("Macierz korelacji")
    plt.show()
    st.pyplot(fig)
    
def turwnak_sprzet_opr_corr():
    st.markdown('### Korelacje - wartość nakładów poniesionych na sprzęt i oprogramowanie w branżach związanych z turystyką')
    
    # plik_wejsciowy =  "C:/Users/Dell/Desktop/streamlit/DATAFRAME/datadzialy_wide_WN_sprzet_opr.xlsx" 
    # df = pd.read_excel(plik_wejsciowy)
    df = pd.read_excel('datadzialy_wide_WN_sprzet_opr.xlsx')
    
    turystyka=df[df["wyszczegolnienie"].isin(['Zakwaterowanie','Wyżywienie','Turystyka'])]
    vector_turystyka = turystyka["wyszczegolnienie"].tolist()
    df_transformed_tur = turystyka.set_index("wyszczegolnienie").T
    df_transformed_tur.columns = vector_turystyka
    df_transformed_tur.index = [2015, 2016, 2017, 2018, 2019, 2020, 2021,2022]
    correlation_matrix = df_transformed_tur.corr()
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
    plt.title("Heatmapa macierzy korelacji")
    plt.show()
    st.pyplot(fig)
    
def turwnak_sprzet_corr():
    st.markdown('### Korelacje - wartość nakładów poniesionych na sprzęt w branżach związanych z turystyką')
    
    # plik_wejsciowy =  "C:/Users/Dell/Desktop/streamlit/DATAFRAME/datadzialy_wide_WN_sprzet_o.xlsx"  
    # df = pd.read_excel(plik_wejsciowy)
    df = pd.read_excel('datadzialy_wide_WN_sprzet_o.xlsx')
    turystyka=df[df["wyszczegolnienie"].isin(['Zakwaterowanie','Wyżywienie','Turystyka'])]
    vector_turystyka = turystyka["wyszczegolnienie"].tolist()
    df_transformed_tur = turystyka.set_index("wyszczegolnienie").T
    df_transformed_tur.columns = vector_turystyka
    df_transformed_tur.index = [2015, 2016, 2017, 2018, 2019, 2020, 2021,2022]
    correlation_matrix = df_transformed_tur.corr()
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
    plt.title("Macierz korelacji")
    plt.show()
    st.pyplot(fig)

def turwnak_sprzet_inf_corr():
    st.markdown('### Korelacje - wartość nakładów poniesionych na sprzęt informatyczny w branżach związanych z turystyką')
    
    # plik_wejsciowy =  "C:/Users/Dell/Desktop/streamlit/DATAFRAME/datadzialy_wide_WN_sprzet_inf.xlsx" 
    # df = pd.read_excel(plik_wejsciowy)
    df = pd.read_excel('datadzialy_wide_WN_sprzet_inf.xlsx')
    
    turystyka=df[df["wyszczegolnienie"].isin(['Zakwaterowanie','Wyżywienie','Turystyka'])]
    vector_turystyka = turystyka["wyszczegolnienie"].tolist()
    df_transformed_tur = turystyka.set_index("wyszczegolnienie").T
    df_transformed_tur.columns = vector_turystyka
    df_transformed_tur.index = [2015, 2016, 2017, 2018, 2019, 2020, 2021,2022]
    correlation_matrix = df_transformed_tur.corr()
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
    plt.title("Macierz korelacji")
    plt.show()
    st.pyplot(fig)
    
def turwnak_sprzet_tele_corr():
    st.markdown('### Korelacje - wartość nakładów poniesionych na sprzęt telekomunikacyjny w branżach związanych z turystyką')
    
    # plik_wejsciowy =  "C:/Users/Dell/Desktop/streamlit/DATAFRAME/datadzialy_wide_WN_sprzet_tele.xlsx" 
    # df = pd.read_excel(plik_wejsciowy)
    df = pd.read_excel('datadzialy_wide_WN_sprzet_tele.xlsx')
    
    turystyka=df[df["wyszczegolnienie"].isin(['Zakwaterowanie','Wyżywienie','Turystyka'])]
    vector_turystyka = turystyka["wyszczegolnienie"].tolist()
    df_transformed_tur = turystyka.set_index("wyszczegolnienie").T
    df_transformed_tur.columns = vector_turystyka
    df_transformed_tur.index = [2015, 2016, 2017, 2018, 2019, 2020, 2021,2022]
    correlation_matrix = df_transformed_tur.corr()
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
    plt.title("Macierz korelacji")
    plt.show()
    st.pyplot(fig)
    
def turwnak_oprog_corr():
    st.markdown('### Korelacje - wartość nakładów poniesionych na oprogramowanie w branżach związanych z turystyką')
    
    # plik_wejsciowy =  "C:/Users/Dell/Desktop/streamlit/DATAFRAME/datadzialy_wide_WN_oprog.xlsx"  
    # df = pd.read_excel(plik_wejsciowy)
    df = pd.read_excel('datadzialy_wide_WN_oprog.xlsx')
    turystyka=df[df["wyszczegolnienie"].isin(['Zakwaterowanie','Wyżywienie','Turystyka'])]
    vector_turystyka = turystyka["wyszczegolnienie"].tolist()
    df_transformed_tur = turystyka.set_index("wyszczegolnienie").T
    df_transformed_tur.columns = vector_turystyka
    df_transformed_tur.index = [2015, 2016, 2017, 2018, 2019, 2020, 2021,2022]
    correlation_matrix = df_transformed_tur.corr()
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
    plt.title("Macierz korelacji")
    plt.show()
    st.pyplot(fig)    
    
def turcorr_pracownicy_section():
    st.sidebar.header("Pracownicy")
    pages = {
        "Przedsiębiorstwa zatrudniające pracowników": turprzed_o_corr,
        "Przedsiębiorstwa z dostępem do internetu": turprzed_di_corr,
        "Pracujący ogółem": turpr_o_corr,
        "Pracownicy z wyższym wykształceniem": turpr_ww_corr,
        "Pracownicy z dostępem do internetu": turpr_di_corr,
        "Pracownicy z urządzeniami przenośnymi": turpr_up_corr,
    }
    selected_page = st.sidebar.selectbox("Wybierz analizę", pages.keys())
    pages[selected_page]()
    
def turprzed_o_corr():
    st.markdown('### Korelacje - ilość przedsiębiorstw w branżach związanych z turystyką')
    
    # plik_wejsciowy =  "C:/Users/Dell/Desktop/streamlit/DATAFRAME/datadzialy_wide_P_przed_o.xlsx"  
    # df = pd.read_excel(plik_wejsciowy)
    df = pd.read_excel('datadzialy_wide_P_przed_o.xlsx')
    
    turystyka=df[df["wyszczegolnienie"].isin(['Zakwaterowanie','Wyżywienie','Turystyka'])]
    vector_turystyka = turystyka["wyszczegolnienie"].tolist()
    df_transformed_tur = turystyka.set_index("wyszczegolnienie").T
    df_transformed_tur.columns = vector_turystyka
    df_transformed_tur.index = [2015, 2016, 2017, 2018, 2019, 2020, 2021]
    correlation_matrix = df_transformed_tur.corr()
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
    plt.title("Macierz korelacji")
    plt.show()
    st.pyplot(fig)
    
def turprzed_di_corr():
    st.markdown('### Korelacje - ilość przedsiębiorstw z dostępem do internetu w branżach związanych z turystyką')
    
    # plik_wejsciowy =  "C:/Users/Dell/Desktop/streamlit/DATAFRAME/datadzialy_wide_P_przed_di.xlsx" 
    # df = pd.read_excel(plik_wejsciowy)
    df = pd.read_excel('datadzialy_wide_P_przed_di.xlsx')
    turystyka=df[df["wyszczegolnienie"].isin(['Zakwaterowanie','Wyżywienie','Turystyka'])]
    vector_turystyka = turystyka["wyszczegolnienie"].tolist()
    df_transformed_tur = turystyka.set_index("wyszczegolnienie").T
    df_transformed_tur.columns = vector_turystyka
    df_transformed_tur.index = [2015, 2016, 2017, 2018, 2019, 2020, 2021]
    correlation_matrix = df_transformed_tur.corr()
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
    plt.title("Macierz korelacji")
    plt.show()
    st.pyplot(fig)
    
def turpr_o_corr():
    st.markdown('### Korelacje - ilość pracujących ogółem w branżach związanych z turystyką')
    
    # plik_wejsciowy =  "C:/Users/Dell/Desktop/streamlit/DATAFRAME/datadzialy_wide_P_pr_o.xlsx"  
    # df = pd.read_excel(plik_wejsciowy)
    df = pd.read_excel('datadzialy_wide_P_pr_o.xlsx')
    
    turystyka=df[df["wyszczegolnienie"].isin(['Zakwaterowanie','Wyżywienie','Turystyka'])]
    vector_turystyka = turystyka["wyszczegolnienie"].tolist()
    df_transformed_tur = turystyka.set_index("wyszczegolnienie").T
    df_transformed_tur.columns = vector_turystyka
    df_transformed_tur.index = [2015, 2016, 2017, 2018, 2019, 2020, 2021]
    correlation_matrix = df_transformed_tur.corr()
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
    plt.title("Heatmapa macierzy korelacji")
    plt.show()
    st.pyplot(fig)
    
def turpr_ww_corr():
    st.markdown('### Korelacje - ilość pracowników z wykształceniem wyższym w branżach związanych z turystyką')
    
    # plik_wejsciowy =  "C:/Users/Dell/Desktop/streamlit/DATAFRAME/datadzialy_wide_P_pr_ww.xlsx" 
    # df = pd.read_excel(plik_wejsciowy)
    df = pd.read_excel('datadzialy_wide_P_pr_ww.xlsx')
    
    turystyka=df[df["wyszczegolnienie"].isin(['Zakwaterowanie','Wyżywienie','Turystyka'])]
    vector_turystyka = turystyka["wyszczegolnienie"].tolist()
    df_transformed_tur = turystyka.set_index("wyszczegolnienie").T
    df_transformed_tur.columns = vector_turystyka
    df_transformed_tur.index = [2015, 2016, 2017, 2018, 2019, 2020, 2021]
    correlation_matrix = df_transformed_tur.corr()
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
    plt.title("Macierz korelacji")
    plt.show()
    st.pyplot(fig)
    
def turpr_di_corr():
    st.markdown('### Korelacje - ilość pracowników z dostępem do internetu w branżach związanych z turystyką')
    
    # plik_wejsciowy =  "C:/Users/Dell/Desktop/streamlit/DATAFRAME/datadzialy_wide_P_pr_di.xlsx" 
    # df = pd.read_excel(plik_wejsciowy)
    df = pd.read_excel('datadzialy_wide_P_pr_di.xlsx')
    
    turystyka=df[df["wyszczegolnienie"].isin(['Zakwaterowanie','Wyżywienie','Turystyka'])]
    vector_turystyka = turystyka["wyszczegolnienie"].tolist()
    df_transformed_tur = turystyka.set_index("wyszczegolnienie").T
    df_transformed_tur.columns = vector_turystyka
    df_transformed_tur.index = [2015, 2016, 2017, 2018, 2019, 2020, 2021]
    correlation_matrix = df_transformed_tur.corr()
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
    plt.title("Macierz korelacji")
    plt.show()
    st.pyplot(fig)
    
def turpr_up_corr():
    st.markdown('### Korelacje - ilość pracowników dysponująca urządzeniami przenośnymi w branżach związanych z turystyką')
    
    # plik_wejsciowy =  "C:/Users/Dell/Desktop/streamlit/DATAFRAME/datadzialy_wide_P_pr_up.xlsx"  
    # df = pd.read_excel(plik_wejsciowy)
    df = pd.read_excel('datadzialy_wide_P_pr_up.xlsx')
    turystyka=df[df["wyszczegolnienie"].isin(['Zakwaterowanie','Wyżywienie','Turystyka'])]
    vector_turystyka = turystyka["wyszczegolnienie"].tolist()
    df_transformed_tur = turystyka.set_index("wyszczegolnienie").T
    df_transformed_tur.columns = vector_turystyka
    df_transformed_tur.index = [2015, 2016, 2017, 2018, 2019, 2020, 2021]
    correlation_matrix = df_transformed_tur.corr()
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
    plt.title("Macierz korelacji")
    plt.show()
    st.pyplot(fig)
    
def turcorr_rodzaje_polaczen_section():
    st.sidebar.header("Rodzaje Połączeń")
    pages = {
        "szerokopasmowy  dostęp do Internetu": turrodz_pol_szer_corr,
        "szerokopasmowy  dostęp do Internetu poprzez łącze DSL": turrodz_pol_szerDSL_corr,
    }
    selected_page = st.sidebar.selectbox("Wybierz analizę", pages.keys())
    pages[selected_page]()
    
def turrodz_pol_szer_corr():
    st.markdown('### Korelacje - przedsiębiorstwa z dostępem do internetu poprzez łącze szerokopasmowe w przedsiębiorstwach związanych z turystyką')
    
    # plik_wejsciowy =  "C:/Users/Dell/Desktop/streamlit/DATAFRAME/datadzialy_wide_RP_od_szer_o.xlsx" 
    # df = pd.read_excel(plik_wejsciowy)
    df = pd.read_excel('datadzialy_wide_RP_od_szer_o.xlsx')
    turystyka=df[df["wyszczegolnienie"].isin(['Zakwaterowanie','Wyżywienie','Turystyka'])]
    vector_turystyka = turystyka["wyszczegolnienie"].tolist()
    df_transformed_tur = turystyka.set_index("wyszczegolnienie").T
    df_transformed_tur.columns = vector_turystyka
    df_transformed_tur.index = [2015, 2016, 2017, 2018, 2019, 2020, 2021,2022]
    correlation_matrix = df_transformed_tur.corr()
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
    plt.title("Macierz korelacji")
    plt.show()
    st.pyplot(fig)
    
def turrodz_pol_szerDSL_corr():
    st.markdown('### Korelacje - przedsiębiorstwa z dostępem do internetu poprzez łącze szerokopasmowe wykorzystywane z użyciem technologii DSL w przedsiębiorstwach związanych z turystyką')
    
    # plik_wejsciowy =  "C:/Users/Dell/Desktop/streamlit/DATAFRAME/datadzialy_wide_RP_od_szer_DSL.xlsx"  
    # df = pd.read_excel(plik_wejsciowy)
    df = pd.read_excel('datadzialy_wide_RP_od_szer_DSL.xlsx')
    turystyka=df[df["wyszczegolnienie"].isin(['Zakwaterowanie','Wyżywienie','Turystyka'])]
    vector_turystyka = turystyka["wyszczegolnienie"].tolist()
    df_transformed_tur = turystyka.set_index("wyszczegolnienie").T
    df_transformed_tur.columns = vector_turystyka
    df_transformed_tur.index = [2015, 2016, 2017, 2018, 2019, 2020, 2021,2022]
    correlation_matrix = df_transformed_tur.corr()
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
    plt.title("Macierz korelacji")
    plt.show()
    st.pyplot(fig)
    
def turcorr_predkosc_polaczen_section():
    st.sidebar.header("Prędkość Połączeń")
    pages = {
        "szerokopasmowy  dostęp do Internetu - prędkość połączenia mniej niż 100 Mbit/s": turpred_pol_szer_w_corr,
        "szerokopasmowy  dostęp do Internetu - prędkość połączenia przynajmniej 100 Mbit/s": turpred_pol_szer_s_corr
    }
    selected_page = st.sidebar.selectbox("Wybierz analizę", pages.keys())
    pages[selected_page]()
    
def turpred_pol_szer_w_corr():
    st.markdown('### Korelacje - prędkość połączenia mniej niż 100 Mbit/s w przedsiębiorstwach związanych z turystyką')
    
    # plik_wejsciowy =  "C:/Users/Dell/Desktop/streamlit/DATAFRAME/datadzialy_wide_PP_od_szer_w.xlsx" 
    # df = pd.read_excel(plik_wejsciowy)
    df = pd.read_excel('datadzialy_wide_PP_od_szer_w.xlsx')
    
    turystyka=df[df["wyszczegolnienie"].isin(['Zakwaterowanie','Wyżywienie','Turystyka'])]
    vector_turystyka = turystyka["wyszczegolnienie"].tolist()
    df_transformed_tur = turystyka.set_index("wyszczegolnienie").T
    df_transformed_tur.columns = vector_turystyka
    df_transformed_tur.index = [2015, 2016, 2017, 2018, 2019, 2020, 2021,2022]
    correlation_matrix = df_transformed_tur.corr()
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
    plt.title("Macierz korelacji")
    plt.show()
    st.pyplot(fig)
    
def turpred_pol_szer_s_corr():
    st.markdown('### Korelacje - prędkość połączenia przynajmniej 100 Mbit/s w przedsiębiorstwach związanych z turystyką')
    
    # plik_wejsciowy =  "C:/Users/Dell/Desktop/streamlit/DATAFRAME/datadzialy_wide_PP_od_szer_s.xlsx" 
    # df = pd.read_excel(plik_wejsciowy)
    df = pd.read_excel('datadzialy_wide_PP_od_szer_s.xlsx')
    turystyka=df[df["wyszczegolnienie"].isin(['Zakwaterowanie','Wyżywienie','Turystyka'])]
    vector_turystyka = turystyka["wyszczegolnienie"].tolist()
    df_transformed_tur = turystyka.set_index("wyszczegolnienie").T
    df_transformed_tur.columns = vector_turystyka
    df_transformed_tur.index = [2015, 2016, 2017, 2018, 2019, 2020, 2021,2022]
    correlation_matrix = df_transformed_tur.corr()
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
    plt.title("Macierz korelacji")
    plt.show()
    st.pyplot(fig)

############################KORELACJE - MEDIA #######################################################################################
def korelacje_infor_section():
    selected_subsection = st.sidebar.selectbox("Wybierz kategorię", [
        "Nakłady", "Pracownicy", "Wartość Nakładów", "Rodzaje Połączeń", "Prędkość Połączeń"
    ])
    if selected_subsection == "Nakłady":
        icorr_naklady_section()
    elif selected_subsection == "Pracownicy":
        icorr_pracownicy_section()
    elif selected_subsection == "Wartość Nakładów":
        icorr_wartosc_nakladow_section()
    elif selected_subsection == "Rodzaje Połączeń":
        icorr_rodzaje_polaczen_section()
    elif selected_subsection == "Prędkość Połączeń":
        icorr_predkosc_polaczen_section()
        
def icorr_naklady_section():
    st.sidebar.header("Nakłady")
    pages = {
        "Nakłady na sprzęt/oprogramowanie": inak_sprzet_opr_corr,
        "Nakłady - sprzęt ogółem": inak_sprzet_corr,
        "Nakłady - sprzęt informatyczny": inak_sprzet_inf_corr,
        "Nakłady - sprzęt telekomunikacyjny": inak_sprzet_tele_corr,
        "Nakłady - leasing": inak_leasing_corr,
        "Nakłady - oprogramowanie": inak_oprog_corr,
    }
    selected_page = st.sidebar.selectbox("Wybierz analizę", pages.keys())
    pages[selected_page]()
    
def inak_leasing_corr(): 
    st.markdown('### Korelacje - przedsiębiorstwa, które poniosły nakłady na leasing urządzeń ICT w branżach związanych z mediami')
    
    # plik_wejsciowy =  "C:/Users/Dell/Desktop/streamlit/DATAFRAME/datadzialy_wide_N_leasing.xlsx" 
    # df = pd.read_excel(plik_wejsciowy)
    df = pd.read_excel('datadzialy_wide_N_leasing.xlsx')
    
    informacje=df[df["wyszczegolnienie"].isin(['Informacja i komunikacja','Prasa, radio i telewizja, wytwórnie filmowe i fonograficzne','Informatyka i usługi informacyjne'])]
    vector_informacje = informacje["wyszczegolnienie"].tolist()
    df_transformed_infor = informacje.set_index("wyszczegolnienie").T
    df_transformed_infor.columns = vector_informacje
    df_transformed_infor.index = [2015, 2016, 2017, 2018, 2019, 2020, 2021,2022]
    correlation_matrix = df_transformed_infor.corr()
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
    plt.title("Macierz korelacji")
    plt.show()
    st.pyplot(fig)
    
def inak_sprzet_opr_corr():
    st.markdown('### Korelacje - przedsiębiorstwa, które poniosły nakłady na sprzęt i oprogramowanie w branżach związanych z mediami')
    
    # plik_wejsciowy =  "C:/Users/Dell/Desktop/streamlit/DATAFRAME/datadzialy_wide_N_sprzet_opr.xlsx" 
    # df = pd.read_excel(plik_wejsciowy)
    df = pd.read_excel('datadzialy_wide_N_sprzet_opr.xlsx')
    informacje=df[df["wyszczegolnienie"].isin(['Informacja i komunikacja','Prasa, radio i telewizja, wytwórnie filmowe i fonograficzne','Informatyka i usługi informacyjne'])]
    vector_informacje = informacje["wyszczegolnienie"].tolist()
    df_transformed_infor = informacje.set_index("wyszczegolnienie").T
    df_transformed_infor.columns = vector_informacje
    df_transformed_infor.index = [2015, 2016, 2017, 2018, 2019, 2020, 2021,2022]
    correlation_matrix = df_transformed_infor.corr()
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
    plt.title("Heatmapa macierzy korelacji")
    plt.show()
    st.pyplot(fig)
    
def inak_sprzet_corr():
    st.markdown('### Korelacje - przedsiębiorstwa, które poniosły nakłady na sprzęt ogółem w branżach związanych z mediami')
    
    # plik_wejsciowy =  "C:/Users/Dell/Desktop/streamlit/DATAFRAME/datadzialy_wide_N_sprzet_o.xlsx" 
    # df = pd.read_excel(plik_wejsciowy)
    df = pd.read_excel('datadzialy_wide_N_sprzet_o.xlsx')
    informacje=df[df["wyszczegolnienie"].isin(['Informacja i komunikacja','Prasa, radio i telewizja, wytwórnie filmowe i fonograficzne','Informatyka i usługi informacyjne'])]
    vector_informacje = informacje["wyszczegolnienie"].tolist()
    df_transformed_infor = informacje.set_index("wyszczegolnienie").T
    df_transformed_infor.columns = vector_informacje
    df_transformed_infor.index = [2015, 2016, 2017, 2018, 2019, 2020, 2021,2022]
    correlation_matrix = df_transformed_infor.corr()
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
    plt.title("Macierz korelacji")
    plt.show()
    st.pyplot(fig)

def inak_sprzet_inf_corr():
    st.markdown('### Korelacje - przedsiębiorstwa, które poniosły nakłady na sprzęt informatyczny w branżach związanych z mediami')
    
    # plik_wejsciowy =  "C:/Users/Dell/Desktop/streamlit/DATAFRAME/datadzialy_wide_N_sprzet_inf.xlsx"  
    # df = pd.read_excel(plik_wejsciowy)
    df = pd.read_excel('datadzialy_wide_N_sprzet_inf.xlsx')
    informacje=df[df["wyszczegolnienie"].isin(['Informacja i komunikacja','Prasa, radio i telewizja, wytwórnie filmowe i fonograficzne','Informatyka i usługi informacyjne'])]
    vector_informacje = informacje["wyszczegolnienie"].tolist()
    df_transformed_infor = informacje.set_index("wyszczegolnienie").T
    df_transformed_infor.columns = vector_informacje
    df_transformed_infor.index = [2015, 2016, 2017, 2018, 2019, 2020, 2021,2022]
    correlation_matrix = df_transformed_infor.corr()
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
    plt.title("Macierz korelacji")
    plt.show()
    st.pyplot(fig)
    
def inak_sprzet_tele_corr():
    st.markdown('### Korelacje - przedsiębiorstwa, które poniosły nakłady na sprzęt telekomunikacyjny w branżach związanych z mediami')
    
    # plik_wejsciowy =  "C:/Users/Dell/Desktop/streamlit/DATAFRAME/datadzialy_wide_N_sprzet_tele.xlsx"  
    # df = pd.read_excel(plik_wejsciowy)
    df = pd.read_excel('datadzialy_wide_N_sprzet_tele.xlsx')
    informacje=df[df["wyszczegolnienie"].isin(['Informacja i komunikacja','Prasa, radio i telewizja, wytwórnie filmowe i fonograficzne','Informatyka i usługi informacyjne'])]
    vector_informacje = informacje["wyszczegolnienie"].tolist()
    df_transformed_infor = informacje.set_index("wyszczegolnienie").T
    df_transformed_infor.columns = vector_informacje
    df_transformed_infor.index = [2015, 2016, 2017, 2018, 2019, 2020, 2021,2022]
    correlation_matrix = df_transformed_infor.corr()
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
    plt.title("Macierz korelacji")
    plt.show()
    st.pyplot(fig)
    
def inak_oprog_corr():
    st.markdown('### Korelacje - przedsiębiorstwa, które poniosły nakłady na oprogramowanie w branżach związanych z mediami')
    
    # plik_wejsciowy =  "C:/Users/Dell/Desktop/streamlit/DATAFRAME/datadzialy_wide_N_oprog.xlsx" 
    # df = pd.read_excel(plik_wejsciowy)
    df = pd.read_excel('datadzialy_wide_N_oprog.xlsx')
    informacje=df[df["wyszczegolnienie"].isin(['Informacja i komunikacja','Prasa, radio i telewizja, wytwórnie filmowe i fonograficzne','Informatyka i usługi informacyjne'])]
    vector_informacje = informacje["wyszczegolnienie"].tolist()
    df_transformed_infor = informacje.set_index("wyszczegolnienie").T
    df_transformed_infor.columns = vector_informacje
    df_transformed_infor.index = [2015, 2016, 2017, 2018, 2019, 2020, 2021,2022]
    correlation_matrix = df_transformed_infor.corr()
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
    plt.title("Macierz korelacji")
    plt.show()
    st.pyplot(fig)
    
def icorr_wartosc_nakladow_section():
    st.sidebar.header("Wartość Nakładów")
    pages = {
        "Wartość nakładów - sprzęt/oprogramowanie": iwnak_sprzet_opr_corr,
        "Wartość nakładów - sprzęt": iwnak_sprzet_corr,
        "Wartość nakładów - sprzęt informatyczny": iwnak_sprzet_inf_corr,
        "Wartość nakładów - sprzęt teleinformatyczny": iwnak_sprzet_tele_corr,
        "Wartość nakładów - leasing": iwnak_leasing_corr,
        "Wartość nakładów - oprogramowanie": iwnak_oprog_corr,
    }
    selected_page = st.sidebar.selectbox("Wybierz analizę", pages.keys())
    pages[selected_page]()
    
def iwnak_leasing_corr():
    st.markdown('### Korelacje - wartość nakładów poniesionych na leasing urządzeń ICT w przedsiębiorstwach związanych z mediami')
    
    # plik_wejsciowy =  "C:/Users/Dell/Desktop/streamlit/DATAFRAME/datadzialy_wide_WN_leasing.xlsx"  
    # df = pd.read_excel(plik_wejsciowy)
    df = pd.read_excel('datadzialy_wide_WN_leasing.xlsx')
    informacje=df[df["wyszczegolnienie"].isin(['Informacja i komunikacja','Prasa, radio i telewizja, wytwórnie filmowe i fonograficzne','Informatyka i usługi informacyjne'])]
    vector_informacje = informacje["wyszczegolnienie"].tolist()
    df_transformed_infor = informacje.set_index("wyszczegolnienie").T
    df_transformed_infor.columns = vector_informacje
    df_transformed_infor.index = [2015, 2016, 2017, 2018, 2019, 2020, 2021,2022]
    correlation_matrix = df_transformed_infor.corr()
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
    plt.title("Macierz korelacji")
    plt.show()
    st.pyplot(fig)
    
def iwnak_sprzet_opr_corr():
    st.markdown('### Korelacje - wartość nakładów poniesionych na sprzęt i oprogramowanie w przedsiębiorstwach związanych z mediami')
    
    # plik_wejsciowy =  "C:/Users/Dell/Desktop/streamlit/DATAFRAME/datadzialy_wide_WN_sprzet_opr.xlsx"  
    # df = pd.read_excel(plik_wejsciowy)
    df = pd.read_excel('datadzialy_wide_WN_sprzet_opr.xlsx')
    informacje=df[df["wyszczegolnienie"].isin(['Informacja i komunikacja','Prasa, radio i telewizja, wytwórnie filmowe i fonograficzne','Informatyka i usługi informacyjne'])]
    vector_informacje = informacje["wyszczegolnienie"].tolist()
    df_transformed_infor = informacje.set_index("wyszczegolnienie").T
    df_transformed_infor.columns = vector_informacje
    df_transformed_infor.index = [2015, 2016, 2017, 2018, 2019, 2020, 2021,2022]
    correlation_matrix = df_transformed_infor.corr()
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
    plt.title("Macierz korelacji")
    plt.show()
    st.pyplot(fig)
    
def iwnak_sprzet_corr():
    st.markdown('### Korelacje - wartość nakładów poniesionych na sprzęt ogółem w przedsiębiorstwach związanych z mediami')
    
    # plik_wejsciowy =  "C:/Users/Dell/Desktop/streamlit/DATAFRAME/datadzialy_wide_WN_sprzet_o.xlsx"  
    # df = pd.read_excel(plik_wejsciowy)
    df = pd.read_excel('datadzialy_wide_WN_sprzet_o.xlsx')
    informacje=df[df["wyszczegolnienie"].isin(['Informacja i komunikacja','Prasa, radio i telewizja, wytwórnie filmowe i fonograficzne','Informatyka i usługi informacyjne'])]
    vector_informacje = informacje["wyszczegolnienie"].tolist()
    df_transformed_infor = informacje.set_index("wyszczegolnienie").T
    df_transformed_infor.columns = vector_informacje
    df_transformed_infor.index = [2015, 2016, 2017, 2018, 2019, 2020, 2021,2022]
    correlation_matrix = df_transformed_infor.corr()
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
    plt.title("Macierz korelacji")
    plt.show()
    st.pyplot(fig)

def iwnak_sprzet_inf_corr():
    st.markdown('### Korelacje - wartość nakładów poniesionych na sprzęt informatyczny w przedsiębiorstwach związanych z mediami')
    
    # plik_wejsciowy =  "C:/Users/Dell/Desktop/streamlit/DATAFRAME/datadzialy_wide_WN_sprzet_inf.xlsx"  
    # df = pd.read_excel(plik_wejsciowy)
    df = pd.read_excel('datadzialy_wide_WN_sprzet_inf.xlsx')
    informacje=df[df["wyszczegolnienie"].isin(['Informacja i komunikacja','Prasa, radio i telewizja, wytwórnie filmowe i fonograficzne','Informatyka i usługi informacyjne'])]
    vector_informacje = informacje["wyszczegolnienie"].tolist()
    df_transformed_infor = informacje.set_index("wyszczegolnienie").T
    df_transformed_infor.columns = vector_informacje
    df_transformed_infor.index = [2015, 2016, 2017, 2018, 2019, 2020, 2021,2022]
    correlation_matrix = df_transformed_infor.corr()
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
    plt.title("Macierz korelacji")
    plt.show()
    st.pyplot(fig)
    
def iwnak_sprzet_tele_corr():
    st.markdown('### Korelacje - wartość nakładów poniesionych na sprzęt telekomunikacyjny w przedsiębiorstwach związanych z mediami')
    
    # plik_wejsciowy =  "C:/Users/Dell/Desktop/streamlit/DATAFRAME/datadzialy_wide_WN_sprzet_tele.xlsx" 
    # df = pd.read_excel(plik_wejsciowy)
    df = pd.read_excel('datadzialy_wide_WN_sprzet_tele.xlsx')
    informacje=df[df["wyszczegolnienie"].isin(['Informacja i komunikacja','Prasa, radio i telewizja, wytwórnie filmowe i fonograficzne','Informatyka i usługi informacyjne'])]
    vector_informacje = informacje["wyszczegolnienie"].tolist()
    df_transformed_infor = informacje.set_index("wyszczegolnienie").T
    df_transformed_infor.columns = vector_informacje
    df_transformed_infor.index = [2015, 2016, 2017, 2018, 2019, 2020, 2021,2022]
    correlation_matrix = df_transformed_infor.corr()
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
    plt.title("Macierz korelacji")
    plt.show()
    st.pyplot(fig)
    
def iwnak_oprog_corr():
    st.markdown('### Korelacje - wartość nakładów poniesionych na oprogramowanie w przedsiębiorstwach związanych z mediami')
    
    # plik_wejsciowy =  "C:/Users/Dell/Desktop/streamlit/DATAFRAME/datadzialy_wide_WN_oprog.xlsx"  
    # df = pd.read_excel(plik_wejsciowy)
    df = pd.read_excel('datadzialy_wide_WN_oprog.xlsx')
    informacje=df[df["wyszczegolnienie"].isin(['Informacja i komunikacja','Prasa, radio i telewizja, wytwórnie filmowe i fonograficzne','Informatyka i usługi informacyjne'])]
    vector_informacje = informacje["wyszczegolnienie"].tolist()
    df_transformed_infor = informacje.set_index("wyszczegolnienie").T
    df_transformed_infor.columns = vector_informacje
    df_transformed_infor.index = [2015, 2016, 2017, 2018, 2019, 2020, 2021,2022]
    correlation_matrix = df_transformed_infor.corr()
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
    plt.title("Macierz korelacji")
    plt.show()
    st.pyplot(fig)
    
    
def icorr_pracownicy_section():
    st.sidebar.header("Pracownicy")
    pages = {
        "Przedsiębiorstwa zatrudniające pracowników": iprzed_o_corr,
        "Przedsiębiorstwa z dostępem do internetu": iprzed_di_corr,
        "Pracujący ogółem": tpr_o_corr,
        "Pracownicy z wyższym wykształceniem": ipr_ww_corr,
        "Pracownicy z dostępem do internetu": ipr_di_corr,
        "Pracownicy z urządzeniami przenośnymi": ipr_up_corr,
    }
    selected_page = st.sidebar.selectbox("Wybierz analizę", pages.keys())
    pages[selected_page]()
    
def iprzed_o_corr():
    st.markdown('### Korelacje - ilość przedsiębiorstw związanych z mediami')
    
    # plik_wejsciowy =  "C:/Users/Dell/Desktop/streamlit/DATAFRAME/datadzialy_wide_P_przed_o.xlsx"  
    # df = pd.read_excel(plik_wejsciowy)
    df = pd.read_excel('datadzialy_wide_P_przed_o.xlsx')
    informacje=df[df["wyszczegolnienie"].isin(['Informacja i komunikacja','Prasa, radio i telewizja, wytwórnie filmowe i fonograficzne','Informatyka i usługi informacyjne'])]
    vector_informacje = informacje["wyszczegolnienie"].tolist()
    df_transformed_infor = informacje.set_index("wyszczegolnienie").T
    df_transformed_infor.columns = vector_informacje
    df_transformed_infor.index = [2015, 2016, 2017, 2018, 2019, 2020, 2021]
    correlation_matrix = df_transformed_infor.corr()
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
    plt.title("Macierz korelacji")
    plt.show()
    st.pyplot(fig)
    
def iprzed_di_corr():
    st.markdown('### Korelacje - ilość przedsiębiorstw z dostępem do internetu związanych z mediami')
    
    # plik_wejsciowy =  "C:/Users/Dell/Desktop/streamlit/DATAFRAME/datadzialy_wide_P_przed_di.xlsx"  
    # df = pd.read_excel(plik_wejsciowy)
    df = pd.read_excel('datadzialy_wide_P_przed_di.xlsx')
    informacje=df[df["wyszczegolnienie"].isin(['Informacja i komunikacja','Prasa, radio i telewizja, wytwórnie filmowe i fonograficzne','Informatyka i usługi informacyjne'])]
    vector_informacje = informacje["wyszczegolnienie"].tolist()
    df_transformed_infor = informacje.set_index("wyszczegolnienie").T
    df_transformed_infor.columns = vector_informacje
    df_transformed_infor.index = [2015, 2016, 2017, 2018, 2019, 2020, 2021]
    correlation_matrix = df_transformed_infor.corr()
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
    plt.title("Macierz korelacji")
    plt.show()
    st.pyplot(fig)
    
def ipr_o_corr():
    st.markdown('### Korelacje - ilość pracujących ogółem w przedsiębiorstwach związanych z mediami')
    
    # plik_wejsciowy =  "C:/Users/Dell/Desktop/streamlit/DATAFRAME/datadzialy_wide_P_pr_o.xlsx" 
    # df = pd.read_excel(plik_wejsciowy)
    df = pd.read_excel('datadzialy_wide_P_pr_o.xlsx')
    informacje=df[df["wyszczegolnienie"].isin(['Informacja i komunikacja','Prasa, radio i telewizja, wytwórnie filmowe i fonograficzne','Informatyka i usługi informacyjne'])]
    vector_informacje = informacje["wyszczegolnienie"].tolist()
    df_transformed_infor = informacje.set_index("wyszczegolnienie").T
    df_transformed_infor.columns = vector_informacje
    df_transformed_infor.index = [2015, 2016, 2017, 2018, 2019, 2020, 2021]
    correlation_matrix = df_transformed_infor.corr()
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
    plt.title("Macierz korelacji")
    plt.show()
    st.pyplot(fig)
    
def ipr_ww_corr():
    st.markdown('### Korelacje - ilość pracowników z wyższym wykształceniem w przedsiębiorstwach związanych z mediami')
    
    # plik_wejsciowy =  "C:/Users/Dell/Desktop/streamlit/DATAFRAME/datadzialy_wide_P_pr_ww.xlsx" 
    # df = pd.read_excel(plik_wejsciowy)
    df = pd.read_excel('datadzialy_wide_P_pr_ww.xlsx')
    informacje=df[df["wyszczegolnienie"].isin(['Informacja i komunikacja','Prasa, radio i telewizja, wytwórnie filmowe i fonograficzne','Informatyka i usługi informacyjne'])]
    vector_informacje = informacje["wyszczegolnienie"].tolist()
    df_transformed_infor = informacje.set_index("wyszczegolnienie").T
    df_transformed_infor.columns = vector_informacje
    df_transformed_infor.index = [2015, 2016, 2017, 2018, 2019, 2020, 2021]
    correlation_matrix = df_transformed_infor.corr()
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
    plt.title("Macierz korelacji")
    plt.show()
    st.pyplot(fig)
    
def ipr_di_corr():
    st.markdown('### Korelacje - ilość pracowników z dostępem do internetu w przedsiębiorstwach związanych z mediami')
    
    # plik_wejsciowy =  "C:/Users/Dell/Desktop/streamlit/DATAFRAME/datadzialy_wide_P_pr_di.xlsx" 
    # df = pd.read_excel(plik_wejsciowy)
    df = pd.read_excel('datadzialy_wide_P_pr_di.xlsx')
    informacje=df[df["wyszczegolnienie"].isin(['Informacja i komunikacja','Prasa, radio i telewizja, wytwórnie filmowe i fonograficzne','Informatyka i usługi informacyjne'])]
    vector_informacje = informacje["wyszczegolnienie"].tolist()
    df_transformed_infor = informacje.set_index("wyszczegolnienie").T
    df_transformed_infor.columns = vector_informacje
    df_transformed_infor.index = [2015, 2016, 2017, 2018, 2019, 2020, 2021]
    correlation_matrix = df_transformed_infor.corr()
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
    plt.title("Macierz korelacji")
    plt.show()
    st.pyplot(fig)
    
def ipr_up_corr():
    st.markdown('### Korelacje - ilość pracowników dysponujących urządzeniami przenośnymi w przedsiębiorstwach związanych z mediami')
    
    # plik_wejsciowy =  "C:/Users/Dell/Desktop/streamlit/DATAFRAME/datadzialy_wide_P_pr_up.xlsx"  
    # df = pd.read_excel(plik_wejsciowy)
    df = pd.read_excel('datadzialy_wide_P_pr_up.xlsx')
    informacje=df[df["wyszczegolnienie"].isin(['Informacja i komunikacja','Prasa, radio i telewizja, wytwórnie filmowe i fonograficzne','Informatyka i usługi informacyjne'])]
    vector_informacje = informacje["wyszczegolnienie"].tolist()
    df_transformed_infor = informacje.set_index("wyszczegolnienie").T
    df_transformed_infor.columns = vector_informacje
    df_transformed_infor.index = [2015, 2016, 2017, 2018, 2019, 2020, 2021]
    correlation_matrix = df_transformed_infor.corr()
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
    plt.title("Macierz korelacji")
    plt.show()
    st.pyplot(fig)
    
def icorr_rodzaje_polaczen_section():
    st.sidebar.header("Rodzaje Połączeń")
    pages = {
        "szerokopasmowy  dostęp do Internetu": irodz_pol_szer_corr,
        "szerokopasmowy  dostęp do Internetu poprzez łącze DSL": irodz_pol_szerDSL_corr,
    }
    selected_page = st.sidebar.selectbox("Wybierz analizę", pages.keys())
    pages[selected_page]()
    
def irodz_pol_szer_corr():
    st.markdown('### Korelacje - przedsiębiorstwa z dostępem do internetu poprzez łącze szerokopasmowe w branżach związanych z mediami')
    
    # plik_wejsciowy =  "C:/Users/Dell/Desktop/streamlit/DATAFRAME/datadzialy_wide_RP_od_szer_o.xlsx"  
    # df = pd.read_excel(plik_wejsciowy)
    df = pd.read_excel('datadzialy_wide_RP_od_szer_o.xlsx')
    informacje=df[df["wyszczegolnienie"].isin(['Informacja i komunikacja','Prasa, radio i telewizja, wytwórnie filmowe i fonograficzne','Informatyka i usługi informacyjne'])]
    vector_informacje = informacje["wyszczegolnienie"].tolist()
    df_transformed_infor = informacje.set_index("wyszczegolnienie").T
    df_transformed_infor.columns = vector_informacje
    df_transformed_infor.index = [2015, 2016, 2017, 2018, 2019, 2020, 2021,2022]
    correlation_matrix = df_transformed_infor.corr()
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
    plt.title("Macierz korelacji")
    plt.show()
    st.pyplot(fig)
    
def irodz_pol_szerDSL_corr():
    st.markdown('### Korelacje - przedsiębiorstwa z dostępem do internetu poprzez łącze szerokopasmowe wykorzystywane z użyciem technologii DSL w branżach związanych z mediami')
    
    # plik_wejsciowy =  "C:/Users/Dell/Desktop/streamlit/DATAFRAME/datadzialy_wide_RP_od_szer_DSL.xlsx" 
    # df = pd.read_excel(plik_wejsciowy)
    df = pd.read_excel('datadzialy_wide_RP_od_szer_DSL.xlsx')
    informacje=df[df["wyszczegolnienie"].isin(['Informacja i komunikacja','Prasa, radio i telewizja, wytwórnie filmowe i fonograficzne','Informatyka i usługi informacyjne'])]
    vector_informacje = informacje["wyszczegolnienie"].tolist()
    df_transformed_infor = informacje.set_index("wyszczegolnienie").T
    df_transformed_infor.columns = vector_informacje
    df_transformed_infor.index = [2015, 2016, 2017, 2018, 2019, 2020, 2021,2022]
    correlation_matrix = df_transformed_infor.corr()
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
    plt.title("Macierz korelacji")
    plt.show()
    st.pyplot(fig)
    
def icorr_predkosc_polaczen_section():
    st.sidebar.header("Prędkość Połączeń")
    pages = {
        "szerokopasmowy  dostęp do Internetu - prędkość połączenia mniej niż 100 Mbit/s": ipred_pol_szer_w_corr,
        "szerokopasmowy  dostęp do Internetu - prędkość połączenia przynajmniej 100 Mbit/s": ipred_pol_szer_s_corr
    }
    selected_page = st.sidebar.selectbox("Wybierz analizę", pages.keys())
    pages[selected_page]()
    
def ipred_pol_szer_w_corr():
    st.markdown('### Korelacje - prędkość połączenia mniej niż 100 Mbit/s w przedsiębiorstwach związanych z mediami')
    
    # plik_wejsciowy =  "C:/Users/Dell/Desktop/streamlit/DATAFRAME/datadzialy_wide_PP_od_szer_w.xlsx"  
    # df = pd.read_excel(plik_wejsciowy)
    df = pd.read_excel('datadzialy_wide_PP_od_szer_w.xlsx')
    informacje=df[df["wyszczegolnienie"].isin(['Informacja i komunikacja','Prasa, radio i telewizja, wytwórnie filmowe i fonograficzne','Informatyka i usługi informacyjne'])]
    vector_informacje = informacje["wyszczegolnienie"].tolist()
    df_transformed_infor = informacje.set_index("wyszczegolnienie").T
    df_transformed_infor.columns = vector_informacje
    df_transformed_infor.index = [2015, 2016, 2017, 2018, 2019, 2020, 2021,2022]
    correlation_matrix = df_transformed_infor.corr()
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
    plt.title("Macierz korelacji")
    plt.show()
    st.pyplot(fig)
    
def ipred_pol_szer_s_corr():
    st.markdown('### Korelacje - prędkość połączenia przynajmniej 100 Mbit/s w przedsiębiorstwach związanych z mediami')
    
    # plik_wejsciowy =  "C:/Users/Dell/Desktop/streamlit/DATAFRAME/datadzialy_wide_PP_od_szer_s.xlsx" 
    # df = pd.read_excel(plik_wejsciowy)
    df = pd.read_excel('datadzialy_wide_PP_od_szer_s.xlsx')
    informacje=df[df["wyszczegolnienie"].isin(['Informacja i komunikacja','Prasa, radio i telewizja, wytwórnie filmowe i fonograficzne','Informatyka i usługi informacyjne'])]
    vector_informacje = informacje["wyszczegolnienie"].tolist()
    df_transformed_infor = informacje.set_index("wyszczegolnienie").T
    df_transformed_infor.columns = vector_informacje
    df_transformed_infor.index = [2015, 2016, 2017, 2018, 2019, 2020, 2021,2022]
    correlation_matrix = df_transformed_infor.corr()
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
    plt.title("Macierz korelacji")
    plt.show()
    st.pyplot(fig)

############################KORELACJE - TECHNOLOGIA######################################################################################
def korelacje_tech_section():
    selected_subsection = st.sidebar.selectbox("Wybierz kategorię", [
        "Nakłady", "Pracownicy", "Wartość Nakładów", "Rodzaje Połączeń", "Prędkość Połączeń"
    ])
    if selected_subsection == "Nakłady":
        tcorr_naklady_section()
    elif selected_subsection == "Pracownicy":
        tcorr_pracownicy_section()
    elif selected_subsection == "Wartość Nakładów":
        tcorr_wartosc_nakladow_section()
    elif selected_subsection == "Rodzaje Połączeń":
        tcorr_rodzaje_polaczen_section()
    elif selected_subsection == "Prędkość Połączeń":
        tcorr_predkosc_polaczen_section()
        
def tcorr_naklady_section():
    st.sidebar.header("Nakłady")
    pages = {
        "Nakłady na sprzęt/oprogramowanie": tnak_sprzet_opr_corr,
        "Nakłady - sprzęt ogółem": tnak_sprzet_corr,
        "Nakłady - sprzęt informatyczny": tnak_sprzet_inf_corr,
        "Nakłady - sprzęt telekomunikacyjny": tnak_sprzet_tele_corr,
        "Nakłady - leasing": tnak_leasing_corr,
        "Nakłady - oprogramowanie": tnak_oprog_corr,
    }
    selected_page = st.sidebar.selectbox("Wybierz analizę", pages.keys())
    pages[selected_page]()
    
def tnak_leasing_corr():
    st.markdown('### Korelacje - przedsiębiorstwa, które poniosły nakłady na leasing urządzeń ICT w branżach związanych z technologiami informacyjno-komunikacyjnymi')
    
    # plik_wejsciowy =  "C:/Users/Dell/Desktop/streamlit/DATAFRAME/datadzialy_wide_N_leasing.xlsx"  
    # df = pd.read_excel(plik_wejsciowy)
    df = pd.read_excel('datadzialy_wide_N_leasing.xlsx')
    technologie=df[df["wyszczegolnienie"].isin(['Informatyka i usługi informacyjne', 'Produkcja komputerów, wyrobów elektronicznych i optycznych','Działalność związana z oprogramowaniem', 'Działalność związana z doradztwem w zakresie informatyki','Telekomunikacja', 'Informatyka i usługi informacyjne','Sprzedaż hurtowa narzędzi technologii informacyjnej i komunikacyjnej','Handel detaliczny','Badania naukowe i prace rozwojowe','Naprawa i konserwacja komputerów i sprzętu komunikacyjnego','Działalność związana z zarządzaniem urządzeniami informatycznymi'])]
    vector_technologie = technologie["wyszczegolnienie"].tolist()
    df_transformed_tech = technologie.set_index("wyszczegolnienie").T
    df_transformed_tech.columns = vector_technologie
    df_transformed_tech.index = [2015, 2016, 2017, 2018, 2019, 2020, 2021,2022]
    correlation_matrix = df_transformed_tech.corr()
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
    plt.title("Macierz korelacji")
    plt.show()
    st.pyplot(fig)
    
def tnak_sprzet_opr_corr():
    st.markdown('### Korelacje - przedsiębiorstwa, które poniosły nakłady na sprzęt i oprogramowanie w branżach związanych z technologiami informacyjno-komunikacyjnymi')
    
    # plik_wejsciowy =  "C:/Users/Dell/Desktop/streamlit/DATAFRAME/datadzialy_wide_N_sprzet_opr.xlsx" 
    # df = pd.read_excel(plik_wejsciowy)
    df = pd.read_excel('datadzialy_wide_N_sprzet_opr.xlsx')
    technologie=df[df["wyszczegolnienie"].isin(['Informatyka i usługi informacyjne', 'Produkcja komputerów, wyrobów elektronicznych i optycznych','Działalność związana z oprogramowaniem', 'Działalność związana z doradztwem w zakresie informatyki','Telekomunikacja', 'Informatyka i usługi informacyjne','Sprzedaż hurtowa narzędzi technologii informacyjnej i komunikacyjnej','Handel detaliczny','Badania naukowe i prace rozwojowe','Naprawa i konserwacja komputerów i sprzętu komunikacyjnego','Działalność związana z zarządzaniem urządzeniami informatycznymi'])]
    vector_technologie = technologie["wyszczegolnienie"].tolist()
    df_transformed_tech = technologie.set_index("wyszczegolnienie").T
    df_transformed_tech.columns = vector_technologie
    df_transformed_tech.index = [2015, 2016, 2017, 2018, 2019, 2020, 2021,2022]
    correlation_matrix = df_transformed_tech.corr()
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
    plt.title("Macierz korelacji")
    plt.show()
    st.pyplot(fig)
    
def tnak_sprzet_corr():
    st.markdown('### Korelacje - przedsiębiorstwa, które poniosły nakłady na sprzęt w branżach związanych z technologiami informacyjno-komunikacyjnymi')
    
    # plik_wejsciowy =  "C:/Users/Dell/Desktop/streamlit/DATAFRAME/datadzialy_wide_N_sprzet_o.xlsx"  
    # df = pd.read_excel(plik_wejsciowy)
    df = pd.read_excel('datadzialy_wide_N_sprzet_o.xlsx')
    technologie=df[df["wyszczegolnienie"].isin(['Informatyka i usługi informacyjne', 'Produkcja komputerów, wyrobów elektronicznych i optycznych','Działalność związana z oprogramowaniem', 'Działalność związana z doradztwem w zakresie informatyki','Telekomunikacja', 'Informatyka i usługi informacyjne','Sprzedaż hurtowa narzędzi technologii informacyjnej i komunikacyjnej','Handel detaliczny','Badania naukowe i prace rozwojowe','Naprawa i konserwacja komputerów i sprzętu komunikacyjnego','Działalność związana z zarządzaniem urządzeniami informatycznymi'])]
    vector_technologie = technologie["wyszczegolnienie"].tolist()
    df_transformed_tech = technologie.set_index("wyszczegolnienie").T
    df_transformed_tech.columns = vector_technologie
    df_transformed_tech.index = [2015, 2016, 2017, 2018, 2019, 2020, 2021,2022]
    correlation_matrix = df_transformed_tech.corr()
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
    plt.title("Macierz korelacji")
    plt.show()
    st.pyplot(fig)

def tnak_sprzet_inf_corr():
    st.markdown('### Korelacje - przedsiębiorstwa, które poniosły nakłady na sprzęt informatyczny w branżach związanych z technologiami informacyjno-komunikacyjnymi')
    
    # plik_wejsciowy =  "C:/Users/Dell/Desktop/streamlit/DATAFRAME/datadzialy_wide_N_sprzet_inf.xlsx"  
    # df = pd.read_excel(plik_wejsciowy)
    df = pd.read_excel('datadzialy_wide_N_sprzet_inf.xlsx')
    
    technologie=df[df["wyszczegolnienie"].isin(['Informatyka i usługi informacyjne', 'Produkcja komputerów, wyrobów elektronicznych i optycznych','Działalność związana z oprogramowaniem', 'Działalność związana z doradztwem w zakresie informatyki','Telekomunikacja', 'Informatyka i usługi informacyjne','Sprzedaż hurtowa narzędzi technologii informacyjnej i komunikacyjnej','Handel detaliczny','Badania naukowe i prace rozwojowe','Naprawa i konserwacja komputerów i sprzętu komunikacyjnego','Działalność związana z zarządzaniem urządzeniami informatycznymi'])]
    vector_technologie = technologie["wyszczegolnienie"].tolist()
    df_transformed_tech = technologie.set_index("wyszczegolnienie").T
    df_transformed_tech.columns = vector_technologie
    df_transformed_tech.index = [2015, 2016, 2017, 2018, 2019, 2020, 2021,2022]
    correlation_matrix = df_transformed_tech.corr()
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
    plt.title("Macierz korelacji")
    plt.show()
    st.pyplot(fig)
    
def tnak_sprzet_tele_corr():
    st.markdown('### Korelacje - przedsiębiorstwa, które poniosły nakłady na sprzęt telekomunikacyjny w branżach związanych z technologiami informacyjno-komunikacyjnymi')
    
    # plik_wejsciowy =  "C:/Users/Dell/Desktop/streamlit/DATAFRAME/datadzialy_wide_N_sprzet_tele.xlsx" 
    # df = pd.read_excel(plik_wejsciowy)
    df = pd.read_excel('datadzialy_wide_N_sprzet_tele.xlsx')
    technologie=df[df["wyszczegolnienie"].isin(['Informatyka i usługi informacyjne', 'Produkcja komputerów, wyrobów elektronicznych i optycznych','Działalność związana z oprogramowaniem', 'Działalność związana z doradztwem w zakresie informatyki','Telekomunikacja', 'Informatyka i usługi informacyjne','Sprzedaż hurtowa narzędzi technologii informacyjnej i komunikacyjnej','Handel detaliczny','Badania naukowe i prace rozwojowe','Naprawa i konserwacja komputerów i sprzętu komunikacyjnego','Działalność związana z zarządzaniem urządzeniami informatycznymi'])]
    vector_technologie = technologie["wyszczegolnienie"].tolist()
    df_transformed_tech = technologie.set_index("wyszczegolnienie").T
    df_transformed_tech.columns = vector_technologie
    df_transformed_tech.index = [2015, 2016, 2017, 2018, 2019, 2020, 2021,2022]
    correlation_matrix = df_transformed_tech.corr()
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
    plt.title("Macierz korelacji")
    plt.show()
    st.pyplot(fig)
    
def tnak_oprog_corr():
    st.markdown('### Korelacje - przedsiębiorstwa, które poniosły nakłady na oprogramowanie w branżach związanych z technologiami informacyjno-komunikacyjnymi')
    
    # plik_wejsciowy =  "C:/Users/Dell/Desktop/streamlit/DATAFRAME/datadzialy_wide_N_oprog.xlsx"  
    # df = pd.read_excel(plik_wejsciowy)
    df = pd.read_excel('datadzialy_wide_N_oprog.xlsx')
    technologie=df[df["wyszczegolnienie"].isin(['Informatyka i usługi informacyjne', 'Produkcja komputerów, wyrobów elektronicznych i optycznych','Działalność związana z oprogramowaniem', 'Działalność związana z doradztwem w zakresie informatyki','Telekomunikacja', 'Informatyka i usługi informacyjne','Sprzedaż hurtowa narzędzi technologii informacyjnej i komunikacyjnej','Handel detaliczny','Badania naukowe i prace rozwojowe','Naprawa i konserwacja komputerów i sprzętu komunikacyjnego','Działalność związana z zarządzaniem urządzeniami informatycznymi'])]
    vector_technologie = technologie["wyszczegolnienie"].tolist()
    df_transformed_tech = technologie.set_index("wyszczegolnienie").T
    df_transformed_tech.columns = vector_technologie
    df_transformed_tech.index = [2015, 2016, 2017, 2018, 2019, 2020, 2021,2022]
    correlation_matrix = df_transformed_tech.corr()
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
    plt.title("Macierz korelacji")
    plt.show()
    st.pyplot(fig)
    
def tcorr_wartosc_nakladow_section():
    st.sidebar.header("Wartość Nakładów")
    pages = {
        "Wartość nakładów - sprzęt/oprogramowanie": twnak_sprzet_opr_corr,
        "Wartość nakładów - sprzęt": twnak_sprzet_corr,
        "Wartość nakładów - sprzęt informatyczny": twnak_sprzet_inf_corr,
        "Wartość nakładów - sprzęt teleinformatyczny": twnak_sprzet_tele_corr,
        "Wartość nakładów - leasing": twnak_leasing_corr,
        "Wartość nakładów - oprogramowanie": twnak_oprog_corr,
    }
    selected_page = st.sidebar.selectbox("Wybierz analizę", pages.keys())
    pages[selected_page]()
    
def twnak_leasing_corr():
    st.markdown('### Korelacje - wartość nakładów poniesionych na leasing urządzeń ICT w przedsiębiorstwach związanych z technologiami informacyjno-komunikacyjnymi')
    
    # plik_wejsciowy =  "C:/Users/Dell/Desktop/streamlit/DATAFRAME/datadzialy_wide_WN_leasing.xlsx"  
    # df = pd.read_excel(plik_wejsciowy)
    df = pd.read_excel('datadzialy_wide_WN_leasing.xlsx')
    technologie=df[df["wyszczegolnienie"].isin(['Informatyka i usługi informacyjne', 'Produkcja komputerów, wyrobów elektronicznych i optycznych','Działalność związana z oprogramowaniem', 'Działalność związana z doradztwem w zakresie informatyki','Telekomunikacja', 'Informatyka i usługi informacyjne','Sprzedaż hurtowa narzędzi technologii informacyjnej i komunikacyjnej','Handel detaliczny','Badania naukowe i prace rozwojowe','Naprawa i konserwacja komputerów i sprzętu komunikacyjnego','Działalność związana z zarządzaniem urządzeniami informatycznymi'])]
    vector_technologie = technologie["wyszczegolnienie"].tolist()
    df_transformed_tech = technologie.set_index("wyszczegolnienie").T
    df_transformed_tech.columns = vector_technologie
    df_transformed_tech.index = [2015, 2016, 2017, 2018, 2019, 2020, 2021,2022]
    correlation_matrix = df_transformed_tech.corr()
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
    plt.title("Macierz korelacji")
    plt.show()
    st.pyplot(fig)
    
def twnak_sprzet_opr_corr():
    st.markdown('### Korelacje - wartość nakładów poniesionych na sprzęt i oprogramowanie w przedsiębiorstwach związanych z technologiami informacyjno-komunikacyjnymi')
    
    # plik_wejsciowy =  "C:/Users/Dell/Desktop/streamlit/DATAFRAME/datadzialy_wide_WN_sprzet_opr.xlsx" 
    # df = pd.read_excel(plik_wejsciowy)
    df = pd.read_excel('datadzialy_wide_WN_sprzet_opr.xlsx')
    
    technologie=df[df["wyszczegolnienie"].isin(['Informatyka i usługi informacyjne', 'Produkcja komputerów, wyrobów elektronicznych i optycznych','Działalność związana z oprogramowaniem', 'Działalność związana z doradztwem w zakresie informatyki','Telekomunikacja', 'Informatyka i usługi informacyjne','Sprzedaż hurtowa narzędzi technologii informacyjnej i komunikacyjnej','Handel detaliczny','Badania naukowe i prace rozwojowe','Naprawa i konserwacja komputerów i sprzętu komunikacyjnego','Działalność związana z zarządzaniem urządzeniami informatycznymi'])]
    vector_technologie = technologie["wyszczegolnienie"].tolist()
    df_transformed_tech = technologie.set_index("wyszczegolnienie").T
    df_transformed_tech.columns = vector_technologie
    df_transformed_tech.index = [2015, 2016, 2017, 2018, 2019, 2020, 2021,2022]
    correlation_matrix = df_transformed_tech.corr()
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
    plt.title("Macierz korelacji")
    plt.show()
    st.pyplot(fig)
    
def twnak_sprzet_corr():
    st.markdown('### Korelacje - wartość nakładów poniesionych na sprzęt ogółem w przedsiębiorstwach związanych z technologiami informacyjno-komunikacyjnymi')
    
    # plik_wejsciowy =  "C:/Users/Dell/Desktop/streamlit/DATAFRAME/datadzialy_wide_WN_sprzet_o.xlsx" 
    # df = pd.read_excel(plik_wejsciowy)
    df = pd.read_excel('datadzialy_wide_WN_sprzet_o.xlsx')
    technologie=df[df["wyszczegolnienie"].isin(['Informatyka i usługi informacyjne', 'Produkcja komputerów, wyrobów elektronicznych i optycznych','Działalność związana z oprogramowaniem', 'Działalność związana z doradztwem w zakresie informatyki','Telekomunikacja', 'Informatyka i usługi informacyjne','Sprzedaż hurtowa narzędzi technologii informacyjnej i komunikacyjnej','Handel detaliczny','Badania naukowe i prace rozwojowe','Naprawa i konserwacja komputerów i sprzętu komunikacyjnego','Działalność związana z zarządzaniem urządzeniami informatycznymi'])]
    vector_technologie = technologie["wyszczegolnienie"].tolist()
    df_transformed_tech = technologie.set_index("wyszczegolnienie").T
    df_transformed_tech.columns = vector_technologie
    df_transformed_tech.index = [2015, 2016, 2017, 2018, 2019, 2020, 2021,2022]
    correlation_matrix = df_transformed_tech.corr()
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
    plt.title("Macierz korelacji")
    plt.show()
    st.pyplot(fig)

def twnak_sprzet_inf_corr():
    st.markdown('### Korelacje - wartość nakładów poniesionych na sprzęt informatyczny w przedsiębiorstwach związanych z technologiami informacyjno-komunikacyjnymi')
    
    # plik_wejsciowy =  "C:/Users/Dell/Desktop/streamlit/DATAFRAME/datadzialy_wide_WN_sprzet_inf.xlsx" 
    # df = pd.read_excel(plik_wejsciowy)
    df = pd.read_excel('datadzialy_wide_WN_sprzet_inf.xlsx')
    
    technologie=df[df["wyszczegolnienie"].isin(['Informatyka i usługi informacyjne', 'Produkcja komputerów, wyrobów elektronicznych i optycznych','Działalność związana z oprogramowaniem', 'Działalność związana z doradztwem w zakresie informatyki','Telekomunikacja', 'Informatyka i usługi informacyjne','Sprzedaż hurtowa narzędzi technologii informacyjnej i komunikacyjnej','Handel detaliczny','Badania naukowe i prace rozwojowe','Naprawa i konserwacja komputerów i sprzętu komunikacyjnego','Działalność związana z zarządzaniem urządzeniami informatycznymi'])]
    vector_technologie = technologie["wyszczegolnienie"].tolist()
    df_transformed_tech = technologie.set_index("wyszczegolnienie").T
    df_transformed_tech.columns = vector_technologie
    df_transformed_tech.index = [2015, 2016, 2017, 2018, 2019, 2020, 2021,2022]
    correlation_matrix = df_transformed_tech.corr()
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
    plt.title("Macierz korelacji")
    plt.show()
    st.pyplot(fig)
    
def twnak_sprzet_tele_corr():
    st.markdown('### Korelacje - wartość nakładów poniesionych na sprzęt telekomunikacyjny w przedsiębiorstwach związanych z technologiami informacyjno-komunikacyjnymi')
    
    # plik_wejsciowy =  "C:/Users/Dell/Desktop/streamlit/DATAFRAME/datadzialy_wide_WN_sprzet_tele.xlsx"  
    # df = pd.read_excel(plik_wejsciowy)
    df = pd.read_excel('datadzialy_wide_WN_sprzet_tele.xlsx')
    technologie=df[df["wyszczegolnienie"].isin(['Informatyka i usługi informacyjne', 'Produkcja komputerów, wyrobów elektronicznych i optycznych','Działalność związana z oprogramowaniem', 'Działalność związana z doradztwem w zakresie informatyki','Telekomunikacja', 'Informatyka i usługi informacyjne','Sprzedaż hurtowa narzędzi technologii informacyjnej i komunikacyjnej','Handel detaliczny','Badania naukowe i prace rozwojowe','Naprawa i konserwacja komputerów i sprzętu komunikacyjnego','Działalność związana z zarządzaniem urządzeniami informatycznymi'])]
    vector_technologie = technologie["wyszczegolnienie"].tolist()
    df_transformed_tech = technologie.set_index("wyszczegolnienie").T
    df_transformed_tech.columns = vector_technologie
    df_transformed_tech.index = [2015, 2016, 2017, 2018, 2019, 2020, 2021,2022]
    correlation_matrix = df_transformed_tech.corr()
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
    plt.title("Macierz korelacji")
    plt.show()
    st.pyplot(fig)
    
def twnak_oprog_corr():
    st.markdown('### Korelacje - wartość nakładów poniesionych na oprogramowanie w przedsiębiorstwach związanych z technologiami informacyjno-komunikacyjnymi')
    
    # plik_wejsciowy =  "C:/Users/Dell/Desktop/streamlit/DATAFRAME/datadzialy_wide_WN_oprog.xlsx" 
    # df = pd.read_excel(plik_wejsciowy)
    df = pd.read_excel('datadzialy_wide_WN_oprog.xlsx')
    technologie=df[df["wyszczegolnienie"].isin(['Informatyka i usługi informacyjne', 'Produkcja komputerów, wyrobów elektronicznych i optycznych','Działalność związana z oprogramowaniem', 'Działalność związana z doradztwem w zakresie informatyki','Telekomunikacja', 'Informatyka i usługi informacyjne','Sprzedaż hurtowa narzędzi technologii informacyjnej i komunikacyjnej','Handel detaliczny','Badania naukowe i prace rozwojowe','Naprawa i konserwacja komputerów i sprzętu komunikacyjnego','Działalność związana z zarządzaniem urządzeniami informatycznymi'])]
    vector_technologie = technologie["wyszczegolnienie"].tolist()
    df_transformed_tech = technologie.set_index("wyszczegolnienie").T
    df_transformed_tech.columns = vector_technologie
    df_transformed_tech.index = [2015, 2016, 2017, 2018, 2019, 2020, 2021,2022]
    correlation_matrix = df_transformed_tech.corr()
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
    plt.title("Macierz korelacji")
    plt.show()
    st.pyplot(fig)    
    
def tcorr_pracownicy_section():
    st.sidebar.header("Pracownicy")
    pages = {
        "Przedsiębiorstwa zatrudniające pracowników": tprzed_o_corr,
        "Przedsiębiorstwa z dostępem do internetu": tprzed_di_corr,
        "Pracujący ogółem": tpr_o_corr,
        "Pracownicy z wyższym wykształceniem": tpr_ww_corr,
        "Pracownicy z dostępem do internetu": tpr_di_corr,
        "Pracownicy z urządzeniami przenośnymi": tpr_up_corr,
    }
    selected_page = st.sidebar.selectbox("Wybierz analizę", pages.keys())
    pages[selected_page]()
    
def tprzed_o_corr():
    st.markdown('### Korelacje - ilość przedsiębiorstw w branżach związanych z technologiami informacyjno-komunikacyjnymi')
    
    # plik_wejsciowy =  "C:/Users/Dell/Desktop/streamlit/DATAFRAME/datadzialy_wide_P_przed_o.xlsx"  
    # df = pd.read_excel(plik_wejsciowy)
    df = pd.read_excel('datadzialy_wide_P_przed_o.xlsx')
    technologie=df[df["wyszczegolnienie"].isin(['Informatyka i usługi informacyjne', 'Produkcja komputerów, wyrobów elektronicznych i optycznych','Działalność związana z oprogramowaniem', 'Działalność związana z doradztwem w zakresie informatyki','Telekomunikacja', 'Informatyka i usługi informacyjne','Sprzedaż hurtowa narzędzi technologii informacyjnej i komunikacyjnej','Handel detaliczny','Badania naukowe i prace rozwojowe','Naprawa i konserwacja komputerów i sprzętu komunikacyjnego','Działalność związana z zarządzaniem urządzeniami informatycznymi'])]
    vector_technologie = technologie["wyszczegolnienie"].tolist()
    df_transformed_tech = technologie.set_index("wyszczegolnienie").T
    df_transformed_tech.columns = vector_technologie
    df_transformed_tech.index = [2015, 2016, 2017, 2018, 2019, 2020, 2021]
    correlation_matrix = df_transformed_tech.corr()
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
    plt.title("Macierz korelacji")
    plt.show()
    st.pyplot(fig)
    
def tprzed_di_corr():
    st.markdown('### Korelacje - ilość przedsiębiorstw z dostępem do internetu w branżach związanych z technologiami informacyjno-komunikacyjnymi')
    
    # plik_wejsciowy =  "C:/Users/Dell/Desktop/streamlit/DATAFRAME/datadzialy_wide_P_przed_di.xlsx"  
    # df = pd.read_excel(plik_wejsciowy)
    df = pd.read_excel('datadzialy_wide_P_przed_di.xlsx')
    technologie=df[df["wyszczegolnienie"].isin(['Informatyka i usługi informacyjne', 'Produkcja komputerów, wyrobów elektronicznych i optycznych','Działalność związana z oprogramowaniem', 'Działalność związana z doradztwem w zakresie informatyki','Telekomunikacja', 'Informatyka i usługi informacyjne','Sprzedaż hurtowa narzędzi technologii informacyjnej i komunikacyjnej','Handel detaliczny','Badania naukowe i prace rozwojowe','Naprawa i konserwacja komputerów i sprzętu komunikacyjnego','Działalność związana z zarządzaniem urządzeniami informatycznymi'])]
    vector_technologie = technologie["wyszczegolnienie"].tolist()
    df_transformed_tech = technologie.set_index("wyszczegolnienie").T
    df_transformed_tech.columns = vector_technologie
    df_transformed_tech.index = [2015, 2016, 2017, 2018, 2019, 2020, 2021]
    correlation_matrix = df_transformed_tech.corr()
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
    plt.title("Macierz korelacji")
    plt.show()
    st.pyplot(fig)
    
def tpr_o_corr():
    st.markdown('### Korelacje - ilość pracujących ogółem w branżach związanych z technologiami informacyjno-komunikacyjnymi')
    
    # plik_wejsciowy =  "C:/Users/Dell/Desktop/streamlit/DATAFRAME/datadzialy_wide_P_pr_o.xlsx"  
    # df = pd.read_excel(plik_wejsciowy)
    df = pd.read_excel('datadzialy_wide_P_pr_o.xlsx')
    technologie=df[df["wyszczegolnienie"].isin(['Informatyka i usługi informacyjne', 'Produkcja komputerów, wyrobów elektronicznych i optycznych','Działalność związana z oprogramowaniem', 'Działalność związana z doradztwem w zakresie informatyki','Telekomunikacja', 'Informatyka i usługi informacyjne','Sprzedaż hurtowa narzędzi technologii informacyjnej i komunikacyjnej','Handel detaliczny','Badania naukowe i prace rozwojowe','Naprawa i konserwacja komputerów i sprzętu komunikacyjnego','Działalność związana z zarządzaniem urządzeniami informatycznymi'])]
    vector_technologie = technologie["wyszczegolnienie"].tolist()
    df_transformed_tech = technologie.set_index("wyszczegolnienie").T
    df_transformed_tech.columns = vector_technologie
    df_transformed_tech.index = [2015, 2016, 2017, 2018, 2019, 2020, 2021]
    correlation_matrix = df_transformed_tech.corr()
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
    plt.title("Macierz korelacji")
    plt.show()
    st.pyplot(fig)
    
def tpr_ww_corr():
    st.markdown('### Korelacje - ilość pracowników z wykształceniem wyższym w branżach związanych z technologiami informacyjno-komunikacyjnymi')
    
    # plik_wejsciowy =  "C:/Users/Dell/Desktop/streamlit/DATAFRAME/datadzialy_wide_P_pr_ww.xlsx" 
    # df = pd.read_excel(plik_wejsciowy)
    df = pd.read_excel('datadzialy_wide_P_pr_ww.xlsx')
    technologie=df[df["wyszczegolnienie"].isin(['Informatyka i usługi informacyjne', 'Produkcja komputerów, wyrobów elektronicznych i optycznych','Działalność związana z oprogramowaniem', 'Działalność związana z doradztwem w zakresie informatyki','Telekomunikacja', 'Informatyka i usługi informacyjne','Sprzedaż hurtowa narzędzi technologii informacyjnej i komunikacyjnej','Handel detaliczny','Badania naukowe i prace rozwojowe','Naprawa i konserwacja komputerów i sprzętu komunikacyjnego','Działalność związana z zarządzaniem urządzeniami informatycznymi'])]
    vector_technologie = technologie["wyszczegolnienie"].tolist()
    df_transformed_tech = technologie.set_index("wyszczegolnienie").T
    df_transformed_tech.columns = vector_technologie
    df_transformed_tech.index = [2015, 2016, 2017, 2018, 2019, 2020, 2021]
    correlation_matrix = df_transformed_tech.corr()
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
    plt.title("Macierz korelacji")
    plt.show()
    st.pyplot(fig)
    
def tpr_di_corr():
    st.markdown('### Korelacje - ilość pracowników z dostępem do internetu w branżach związanych z technologiami informacyjno-komunikacyjnymi')
    
    # plik_wejsciowy =  "C:/Users/Dell/Desktop/streamlit/DATAFRAME/datadzialy_wide_P_pr_di.xlsx" 
    # df = pd.read_excel(plik_wejsciowy)
    df = pd.read_excel('datadzialy_wide_P_pr_di.xlsx')
    technologie=df[df["wyszczegolnienie"].isin(['Informatyka i usługi informacyjne', 'Produkcja komputerów, wyrobów elektronicznych i optycznych','Działalność związana z oprogramowaniem', 'Działalność związana z doradztwem w zakresie informatyki','Telekomunikacja', 'Informatyka i usługi informacyjne','Sprzedaż hurtowa narzędzi technologii informacyjnej i komunikacyjnej','Handel detaliczny','Badania naukowe i prace rozwojowe','Naprawa i konserwacja komputerów i sprzętu komunikacyjnego','Działalność związana z zarządzaniem urządzeniami informatycznymi'])]
    vector_technologie = technologie["wyszczegolnienie"].tolist()
    df_transformed_tech = technologie.set_index("wyszczegolnienie").T
    df_transformed_tech.columns = vector_technologie
    df_transformed_tech.index = [2015, 2016, 2017, 2018, 2019, 2020, 2021]
    correlation_matrix = df_transformed_tech.corr()
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
    plt.title("Macierz korelacji")
    plt.show()
    st.pyplot(fig)
    
def tpr_up_corr():
    st.markdown('### Korelacje - ilość pracowników dysponujących urządzeniami przenośnymi w branżach związanych z technologiami informacyjno-komunikacyjnymi')
    
    # plik_wejsciowy =  "C:/Users/Dell/Desktop/streamlit/DATAFRAME/datadzialy_wide_P_pr_up.xlsx" 
    # df = pd.read_excel(plik_wejsciowy)
    df = pd.read_excel('datadzialy_wide_P_pr_up.xlsx')
    technologie=df[df["wyszczegolnienie"].isin(['Informatyka i usługi informacyjne', 'Produkcja komputerów, wyrobów elektronicznych i optycznych','Działalność związana z oprogramowaniem', 'Działalność związana z doradztwem w zakresie informatyki','Telekomunikacja', 'Informatyka i usługi informacyjne','Sprzedaż hurtowa narzędzi technologii informacyjnej i komunikacyjnej','Handel detaliczny','Badania naukowe i prace rozwojowe','Naprawa i konserwacja komputerów i sprzętu komunikacyjnego','Działalność związana z zarządzaniem urządzeniami informatycznymi'])]
    vector_technologie = technologie["wyszczegolnienie"].tolist()
    df_transformed_tech = technologie.set_index("wyszczegolnienie").T
    df_transformed_tech.columns = vector_technologie
    df_transformed_tech.index = [2015, 2016, 2017, 2018, 2019, 2020, 2021]
    correlation_matrix = df_transformed_tech.corr()
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
    plt.title("Macierz korelacji")
    plt.show()
    st.pyplot(fig)
    
def tcorr_rodzaje_polaczen_section():
    st.sidebar.header("Rodzaje Połączeń")
    pages = {
        "szerokopasmowy  dostęp do Internetu": trodz_pol_szer_corr,
        "szerokopasmowy  dostęp do Internetu poprzez łącze DSL": trodz_pol_szerDSL_corr,
    }
    selected_page = st.sidebar.selectbox("Wybierz analizę", pages.keys())
    pages[selected_page]()
    
def trodz_pol_szer_corr():
    st.markdown('### Korelacje - przedsiębiorstwa z dostępem do internetu poprzez łącze szerokopasmowe w branżach związanych z technologiami informacyjno-komunikacyjnymi')
    
    # plik_wejsciowy =  "C:/Users/Dell/Desktop/streamlit/DATAFRAME/datadzialy_wide_RP_od_szer_o.xlsx" 
    # df = pd.read_excel(plik_wejsciowy)
    df = pd.read_excel('datadzialy_wide_RP_od_szer_o.xlsx')
    technologie=df[df["wyszczegolnienie"].isin(['Informatyka i usługi informacyjne', 'Produkcja komputerów, wyrobów elektronicznych i optycznych','Działalność związana z oprogramowaniem','Telekomunikacja', 'Informatyka i usługi informacyjne','Sprzedaż hurtowa narzędzi technologii informacyjnej i komunikacyjnej','Handel detaliczny','Badania naukowe i prace rozwojowe','Naprawa i konserwacja komputerów i sprzętu komunikacyjnego','Działalność związana z zarządzaniem urządzeniami informatycznymi'])]
    vector_technologie = technologie["wyszczegolnienie"].tolist()
    df_transformed_tech = technologie.set_index("wyszczegolnienie").T
    df_transformed_tech.columns = vector_technologie
    df_transformed_tech.index = [2015, 2016, 2017, 2018, 2019, 2020, 2021,2022]
    correlation_matrix = df_transformed_tech.corr()
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
    plt.title("Macierz korelacji")
    plt.show()
    st.pyplot(fig)
    
def trodz_pol_szerDSL_corr():
    st.markdown('### Korelacje - przedsiębiorstwa z dostępem do internetu poprzez łącze szerokopasmowe wykorzystywane z użyciem technologii DSL w branżach związanych z technologiami informacyjno-komunikacyjnymi')
    
    # plik_wejsciowy =  "C:/Users/Dell/Desktop/streamlit/DATAFRAME/datadzialy_wide_RP_od_szer_DSL.xlsx"  
    # df = pd.read_excel(plik_wejsciowy)
    df = pd.read_excel('datadzialy_wide_RP_od_szer_DSL.xlsx')
    technologie=df[df["wyszczegolnienie"].isin(['Informatyka i usługi informacyjne', 'Produkcja komputerów, wyrobów elektronicznych i optycznych','Działalność związana z oprogramowaniem', 'Działalność związana z doradztwem w zakresie informatyki','Telekomunikacja', 'Informatyka i usługi informacyjne','Sprzedaż hurtowa narzędzi technologii informacyjnej i komunikacyjnej','Handel detaliczny','Badania naukowe i prace rozwojowe','Naprawa i konserwacja komputerów i sprzętu komunikacyjnego','Działalność związana z zarządzaniem urządzeniami informatycznymi'])]
    vector_technologie = technologie["wyszczegolnienie"].tolist()
    df_transformed_tech = technologie.set_index("wyszczegolnienie").T
    df_transformed_tech.columns = vector_technologie
    df_transformed_tech.index = [2015, 2016, 2017, 2018, 2019, 2020, 2021,2022]
    correlation_matrix = df_transformed_tech.corr()
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
    plt.title("Macierz korelacji")
    plt.show()
    st.pyplot(fig)
    
def tcorr_predkosc_polaczen_section():
    st.sidebar.header("Prędkość Połączeń")
    pages = {
        "szerokopasmowy  dostęp do Internetu - prędkość połączenia mniej niż 100 Mbit/s": tpred_pol_szer_w_corr,
        "szerokopasmowy  dostęp do Internetu - prędkość połączenia przynajmniej 100 Mbit/s": tpred_pol_szer_s_corr
    }
    selected_page = st.sidebar.selectbox("Wybierz analizę", pages.keys())
    pages[selected_page]()
    
def tpred_pol_szer_w_corr():
    st.markdown('### Korelacje - prędkość połączenia mniej niż 100 Mbit/s w przedsiębiorstwach związanych z technologiami informacyjno-komunikacyjnymi')
    
    # plik_wejsciowy =  "C:/Users/Dell/Desktop/streamlit/DATAFRAME/datadzialy_wide_PP_od_szer_w.xlsx" 
    # df = pd.read_excel(plik_wejsciowy)
    df = pd.read_excel('datadzialy_wide_PP_od_szer_w.xlsx')
    technologie=df[df["wyszczegolnienie"].isin(['Informatyka i usługi informacyjne', 'Produkcja komputerów, wyrobów elektronicznych i optycznych','Działalność związana z oprogramowaniem', 'Działalność związana z doradztwem w zakresie informatyki','Telekomunikacja', 'Informatyka i usługi informacyjne','Sprzedaż hurtowa narzędzi technologii informacyjnej i komunikacyjnej','Handel detaliczny','Badania naukowe i prace rozwojowe','Naprawa i konserwacja komputerów i sprzętu komunikacyjnego','Działalność związana z zarządzaniem urządzeniami informatycznymi'])]
    vector_technologie = technologie["wyszczegolnienie"].tolist()
    df_transformed_tech = technologie.set_index("wyszczegolnienie").T
    df_transformed_tech.columns = vector_technologie
    df_transformed_tech.index = [2015, 2016, 2017, 2018, 2019, 2020, 2021,2022]
    correlation_matrix = df_transformed_tech.corr()
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
    plt.title("Macierz korelacji")
    plt.show()
    st.pyplot(fig)
    
def tpred_pol_szer_s_corr():
    st.markdown('### Korelacje - prędkość połączenia przynajmniej 100 Mbit/s w przedsiębiorstwach związanych z technologiami informacyjno-komunikacyjnymi')
    
    # plik_wejsciowy =  "C:/Users/Dell/Desktop/streamlit/DATAFRAME/datadzialy_wide_PP_od_szer_s.xlsx"  
    # df = pd.read_excel(plik_wejsciowy)
    df = pd.read_excel('datadzialy_wide_PP_od_szer_s.xlsx')
    technologie=df[df["wyszczegolnienie"].isin(['Informatyka i usługi informacyjne', 'Produkcja komputerów, wyrobów elektronicznych i optycznych','Działalność związana z oprogramowaniem', 'Działalność związana z doradztwem w zakresie informatyki','Telekomunikacja', 'Informatyka i usługi informacyjne','Sprzedaż hurtowa narzędzi technologii informacyjnej i komunikacyjnej','Handel detaliczny','Badania naukowe i prace rozwojowe','Naprawa i konserwacja komputerów i sprzętu komunikacyjnego','Działalność związana z zarządzaniem urządzeniami informatycznymi'])]
    vector_technologie = technologie["wyszczegolnienie"].tolist()
    df_transformed_tech = technologie.set_index("wyszczegolnienie").T
    df_transformed_tech.columns = vector_technologie
    df_transformed_tech.index = [2015, 2016, 2017, 2018, 2019, 2020, 2021,2022]
    correlation_matrix = df_transformed_tech.corr()
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
    plt.title("Macierz korelacji")
    plt.show()
    st.pyplot(fig)
    
##### MAPY ##############################################################################################################################
def mapy_section():
    selected_subsection = st.sidebar.selectbox("Wybierz kategorię", [
        "Nakłady", "Pracownicy", "Wartość Nakładów", "Rodzaje Połączeń", "Prędkość Połączeń"
    ])
    
    if selected_subsection == "Nakłady":
        naklady_section()
    elif selected_subsection == "Pracownicy":
        pracownicy_section()
    elif selected_subsection == "Wartość Nakładów":
        wartosc_nakladow_section()
    elif selected_subsection == "Rodzaje Połączeń":
        rodzaje_polaczen_section()
    elif selected_subsection == "Prędkość Połączeń":
        predkosc_polaczen_section()

def naklady_section():
    st.sidebar.header("Nakłady")
    pages = {
        "Nakłady na sprzęt/oprogramowanie": maps_demo2,
        "Nakłady - sprzęt ogółem": maps_demo3,
        "Nakłady - sprzęt informatyczny": maps_demo4,
        "Nakłady - sprzęt telekomunikacyjny": maps_demo5,
        "Nakłady - leasing": maps_demo6,
        "Nakłady - oprogramowanie": maps_demo7,
    }
    selected_page = st.sidebar.selectbox("Wybierz analizę", pages.keys())
    pages[selected_page]()

def pracownicy_section():
    st.sidebar.header("Pracownicy")
    pages = {
        "Przedsiębiorstwa zatrudniające pracowników": maps_demo_pracownicy,
        "Przedsiębiorstwa z dostępem do internetu": maps_demo2_pracownicy,
        "Pracujący ogółem": maps_demo3_pracownicy,
        "Pracownicy z wyższym wykształceniem": maps_demo4_pracownicy,
        "Pracownicy z dostępem do internetu": maps_demo5_pracownicy,
        "Pracownicy z urządzeniami przenośnymi": maps_demo6_pracownicy,
    }
    selected_page = st.sidebar.selectbox("Wybierz analizę", pages.keys())
    pages[selected_page]()

def wartosc_nakladow_section():
    st.sidebar.header("Wartość Nakładów")
    pages = {
        "Wartość nakładów - sprzęt/oprogramowanie": maps_demo_wart_nakladow,
        "Wartość nakładów - sprzęt": maps_demo2_wart_nakladow,
        "Wartość nakładów - sprzęt informatyczny": maps_demo3_wart_nakladow,
        "Wartość nakładów - sprzęt teleinformatyczny": maps_demo4_wart_nakladow,
        "Wartość nakładów - leasing": maps_demo5_wart_nakladow,
        "Wartość nakładów - oprogramowanie": maps_demo6_wart_nakladow,
    }
    selected_page = st.sidebar.selectbox("Wybierz analizę", pages.keys())
    pages[selected_page]()

def rodzaje_polaczen_section():
    st.sidebar.header("Rodzaje Połączeń")
    pages = {
        "szerokopasmowy  dostęp do Internetu": maps_demo2_rodz_polaczen,
        "szerokopasmowy  dostęp do Internetu poprzez łącze DSL": maps_demo3_rodz_polaczen,
    }
    selected_page = st.sidebar.selectbox("Wybierz analizę", pages.keys())
    pages[selected_page]()

def predkosc_polaczen_section():
    st.sidebar.header("Prędkość Połączeń")
    pages = {
        "szerokopasmowy  dostęp do Internetu - prędkość połączenia mniej niż 100 Mbit/s": maps_demo4_pred_polaczen,
        "szerokopasmowy  dostęp do Internetu - prędkość połączenia przynajmniej 100 Mbit/s": maps_demo5_pred_polaczen
    }
    selected_page = st.sidebar.selectbox("Wybierz analizę", pages.keys())
    pages[selected_page]()


def maps_demo2():
    st.markdown('### Odsetek przedsiębiorstw, które poniosły nakłady na sprzęt i oprogramowanie wśród przedsiębiorstw ogółem')
    try:
        # data_file_path = 'C:/Users/Dell/Desktop/merge-P-nakłady.xlsx'
        # data1 = pd.read_excel(data_file_path)
        data1 = pd.read_excel('merge-P-nakłady.xlsx')
        naklady = pd.DataFrame(data1)
        wojewodztwa_naklady = naklady.iloc[47:63]
        wojewodztwa_naklady['wyszczegolnienie'] = wojewodztwa_naklady['wyszczegolnienie'].str.lower()

        lata = [2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022]
        kolumny1 = ['przed_o15', 'przed_o16', 'przed_o17', 'przed_o18', 'przed_o19', 'przed_o20', 'przed_o21', 'przed_o22']
        kolumny2 = ['sprzet/opr15', 'sprzet/opr16', 'sprzet/opr17', 'sprzet/opr18', 'sprzet/opr19', 'sprzet/opr20', 'sprzet/opr21', 'sprzet/opr22']

        kolumny = [wojewodztwa_naklady[kol2] / wojewodztwa_naklady[kol1] for kol1, kol2 in zip(kolumny1, kolumny2)]
        dane_mapy = {rok: wojewodztwa_naklady[['wyszczegolnienie']].copy().assign(wartosc=kol) for rok, kol in zip(lata, kolumny)}
        
        geojson_url = 'https://raw.githubusercontent.com/ppatrzyk/polska-geojson/master/wojewodztwa/wojewodztwa-min.geojson'
        woj = gpd.read_file(geojson_url).set_index('nazwa')
        
        mapy = {}
    
        for rok in lata:
            dane_mapy[rok]['wartosc'] = pd.to_numeric(dane_mapy[rok]['wartosc'], errors='coerce')
            dane_mapy[rok]['text'] = dane_mapy[rok].apply(lambda row: f"{row['wyszczegolnienie'].capitalize()}: {row['wartosc']}", axis=1)
            mapy[rok] = px.choropleth(dane_mapy[rok], geojson=woj, locations="wyszczegolnienie", color="wartosc",
                                      color_continuous_scale="greens", projection="mercator")
            mapy[rok].update_geos(fitbounds="locations", visible=False)

        year = st.sidebar.radio("Wybierz rok", lata)
        st.subheader(f"Mapa dla roku {year}")
        st.plotly_chart(mapy[year], use_container_width=True)
    
    except FileNotFoundError:
        st.error("Nie znaleziono pliku Excel. Upewnij się, że podana ścieżka jest poprawna.")
        return
    
    st.title('Analiza trendów')
    @st.cache_data
    def get_local_data():
        try:
            df = pd.read_excel('nak_sprzet_opr_wojewodztwa.xlsx', sheet_name=0)
            return df.set_index('Region')
        except Exception as e:
            st.error(f'Błąd wczytywania danych: {e}')
            return pd.DataFrame()

    df = get_local_data()
    if df.empty:
        return
    
    countries = st.multiselect('Wybierz regiony', list(df.index))
    if not countries:
        st.error('Wybierz co najmniej jeden region.')
        return
    
    data = df.loc[countries].T.reset_index().rename(columns={'index': 'year'})
    data = pd.melt(data, id_vars=['year'], var_name='Region', value_name='Wartość')
    data['year'] = pd.to_numeric(data['year'], errors='coerce').dropna().astype(int)

    chart = alt.Chart(data).mark_line().encode(
        x=alt.X('year:O', title='Rok'),
        y=alt.Y('Wartość:Q', title='Wartość'),
        color=alt.Color('Region:N', title='Województwo')
    ).properties(
        title=alt.TitleParams(
            text='Przedsiębiorstwa, które poniosły nakłady na sprzęt i oprogramowanie',
            anchor='middle', fontSize=12))    
    
    st.altair_chart(chart, use_container_width=True)
    
    st.write("### Wskaźnik nachylenia Sena")
    slope_data = [{'Region': region, 'Slope': linregress(data[data['Region'] == region]['year'], data[data['Region'] == region]['Wartość'])[0]} for region in countries]
    st.write(pd.DataFrame(slope_data)) 
    
    st.write("### Test Manna-Kendalla")
    mk_data = []
    for region in countries:
        result = mk.original_test(data[data['Region'] == region]['Wartość'].values)
        mk_data.append({
            'Region': region,
            'Trend': result.trend,
            'p-value': result.p
        })
    st.write(pd.DataFrame(mk_data))   
    
def maps_demo3():
    st.title("Odsetek przedsiębiorstw, które poniosły nakłady na sprzęt wśród przedsiębiorstw ogółem")

    try:
        # data_file_path = 'C:/Users/Dell/Desktop/merge-P-nakłady.xlsx'
        # data1 = pd.read_excel(data_file_path)
        data1 = pd.read_excel('merge-P-nakłady.xlsx')
        naklady = pd.DataFrame(data1)
        wojewodztwa_naklady = naklady.iloc[47:63]
        wojewodztwa_naklady['wyszczegolnienie'] = wojewodztwa_naklady['wyszczegolnienie'].str.lower()

        lata = [2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022]
        kolumny1 = ['przed_o15', 'przed_o16', 'przed_o17', 'przed_o18', 'przed_o19', 'przed_o20', 'przed_o21', 'przed_o22']
        kolumny2 = ['sprzet15', 'sprzet16', 'sprzet17', 'sprzet18', 'sprzet19', 'sprzet20', 'sprzet21','sprzet22']

        kolumny = [wojewodztwa_naklady[kol2] / wojewodztwa_naklady[kol1] for kol1, kol2 in zip(kolumny1, kolumny2)]
        dane_mapy = {rok: wojewodztwa_naklady[['wyszczegolnienie']].copy().assign(wartosc=kol) for rok, kol in zip(lata, kolumny)}
        
        geojson_url = 'https://raw.githubusercontent.com/ppatrzyk/polska-geojson/master/wojewodztwa/wojewodztwa-min.geojson'
        woj = gpd.read_file(geojson_url).set_index('nazwa')
        
        mapy = {}
        for rok in lata:
            dane_mapy[rok]['wartosc'] = pd.to_numeric(dane_mapy[rok]['wartosc'], errors='coerce')
            dane_mapy[rok]['text'] = dane_mapy[rok].apply(lambda row: f"{row['wyszczegolnienie'].capitalize()}: {row['wartosc']}", axis=1)
            mapy[rok] = px.choropleth(dane_mapy[rok], geojson=woj, locations="wyszczegolnienie", color="wartosc",
                                      color_continuous_scale="greens", projection="mercator")
            mapy[rok].update_geos(fitbounds="locations", visible=False)

        year = st.sidebar.radio("Wybierz rok", lata)
        st.subheader(f"Mapa dla roku {year}")
        st.plotly_chart(mapy[year], use_container_width=True)
    
    except FileNotFoundError:
        st.error("Nie znaleziono pliku Excel. Upewnij się, że podana ścieżka jest poprawna.")
        return
    
    st.title('Analiza trendów')
    @st.cache_data
    def get_local_data():
        try:
            df = pd.read_excel('nak_sprzet_wojewodztwa.xlsx', sheet_name=0)
            return df.set_index('Region')
        except Exception as e:
            st.error(f'Błąd wczytywania danych: {e}')
            return pd.DataFrame()

    df = get_local_data()
    if df.empty:
        return
    
    countries = st.multiselect('Wybierz regiony', list(df.index))
    if not countries:
        st.error('Wybierz co najmniej jeden region.')
        return
    
    data = df.loc[countries].T.reset_index().rename(columns={'index': 'year'})
    data = pd.melt(data, id_vars=['year'], var_name='Region', value_name='Wartość')
    data['year'] = pd.to_numeric(data['year'], errors='coerce').dropna().astype(int)

    chart = alt.Chart(data).mark_line().encode(
        x=alt.X('year:O', title='Rok'),
        y=alt.Y('Wartość:Q', title='Wartość'),
        color=alt.Color('Region:N', title='Województwo')
    ).properties(
        title=alt.TitleParams(
            text='Przedsiębiorstwa, które poniosły nakłady na sprzęt',
            anchor='middle', fontSize=12))
    st.altair_chart(chart, use_container_width=True)
    
    st.write("### Wskaźnik nachylenia Sena")
    slope_data = [{'Region': region, 'Slope': linregress(data[data['Region'] == region]['year'], data[data['Region'] == region]['Wartość'])[0]} for region in countries]
    st.write(pd.DataFrame(slope_data))
    
    st.write("### Test Manna-Kendalla")
    mk_data = []
    for region in countries:
        result = mk.original_test(data[data['Region'] == region]['Wartość'].values)
        mk_data.append({
            'Region': region,
            'Trend': result.trend,
            'p-value': result.p
        })
    st.write(pd.DataFrame(mk_data)) 
    
def maps_demo4():
    st.title("Odsetek przedsiębiorstw, które poniosły nakłady na sprzęt informatyczny wśród przedsiębiorstw ogółem")

    try:
        # data_file_path = 'C:/Users/Dell/Desktop/merge-P-nakłady.xlsx'
        # data1 = pd.read_excel(data_file_path)
        data1 = pd.read_excel('merge-P-nakłady.xlsx')
        naklady = pd.DataFrame(data1)
        wojewodztwa_naklady = naklady.iloc[47:63]
        wojewodztwa_naklady['wyszczegolnienie'] = wojewodztwa_naklady['wyszczegolnienie'].str.lower()

        lata = [2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022]
        kolumny1 = ['przed_o15', 'przed_o16', 'przed_o17', 'przed_o18', 'przed_o19', 'przed_o20', 'przed_o21', 'przed_o22']
        kolumny2 = ['sprzet_inf15', 'sprzet_inf16', 'sprzet_inf17', 'sprzet_inf18', 'sprzet_inf19', 'sprzet_inf20', 'sprzet_inf21','sprzet_inf22']

        kolumny = [wojewodztwa_naklady[kol2] / wojewodztwa_naklady[kol1] for kol1, kol2 in zip(kolumny1, kolumny2)]
        dane_mapy = {rok: wojewodztwa_naklady[['wyszczegolnienie']].copy().assign(wartosc=kol) for rok, kol in zip(lata, kolumny)}
        
        geojson_url = 'https://raw.githubusercontent.com/ppatrzyk/polska-geojson/master/wojewodztwa/wojewodztwa-min.geojson'
        woj = gpd.read_file(geojson_url).set_index('nazwa')
        
        mapy = {}
        for rok in lata:
            dane_mapy[rok]['wartosc'] = pd.to_numeric(dane_mapy[rok]['wartosc'], errors='coerce')
            dane_mapy[rok]['text'] = dane_mapy[rok].apply(lambda row: f"{row['wyszczegolnienie'].capitalize()}: {row['wartosc']}", axis=1)
            mapy[rok] = px.choropleth(dane_mapy[rok], geojson=woj, locations="wyszczegolnienie", color="wartosc",
                                      color_continuous_scale="greens", projection="mercator")
            mapy[rok].update_geos(fitbounds="locations", visible=False)

        year = st.sidebar.radio("Wybierz rok", lata)
        st.subheader(f"Mapa dla roku {year}")
        st.plotly_chart(mapy[year], use_container_width=True)
    
    except FileNotFoundError:
        st.error("Nie znaleziono pliku Excel. Upewnij się, że podana ścieżka jest poprawna.")
        return
    
    st.title('Analiza trendów')
    @st.cache_data
    def get_local_data():
        try:
            df = pd.read_excel('nak_sprzet_inf_wojewodztwa.xlsx', sheet_name=0)
            return df.set_index('Region')
        except Exception as e:
            st.error(f'Błąd wczytywania danych: {e}')
            return pd.DataFrame()

    df = get_local_data()
    if df.empty:
        return
    
    countries = st.multiselect('Wybierz regiony', list(df.index))
    if not countries:
        st.error('Wybierz co najmniej jeden region.')
        return
    
    data = df.loc[countries].T.reset_index().rename(columns={'index': 'year'})
    data = pd.melt(data, id_vars=['year'], var_name='Region', value_name='Wartość')
    data['year'] = pd.to_numeric(data['year'], errors='coerce').dropna().astype(int)

    chart = alt.Chart(data).mark_line().encode(
        x=alt.X('year:O', title='Rok'),
        y=alt.Y('Wartość:Q', title='Wartość'),
        color=alt.Color('Region:N', title='Województwo')
    ).properties(
        title=alt.TitleParams(
            text='Przedsiębiorstwa, które poniosły nakłady na sprzęt informatyczny',
            anchor='middle', fontSize=12))
    st.altair_chart(chart, use_container_width=True)
    
    st.write("### Wskaźnik nachylenia Sena")
    slope_data = [{'Region': region, 'Slope': linregress(data[data['Region'] == region]['year'], data[data['Region'] == region]['Wartość'])[0]} for region in countries]
    st.write(pd.DataFrame(slope_data))
             
    st.write("### Test Manna-Kendalla")
    mk_data = []
    for region in countries:
        result = mk.original_test(data[data['Region'] == region]['Wartość'].values)
        mk_data.append({
            'Region': region,
            'Trend': result.trend,
            'p-value': result.p
        })
    st.write(pd.DataFrame(mk_data))    

def maps_demo5():
    st.title("Odsetek przedsiębiorstw, które poniosły nakłady na sprzęt telekomunikacyjny wśród przedsiębiorstw ogółem")

    try:
        # data_file_path = 'C:/Users/Dell/Desktop/merge-P-nakłady.xlsx'
        # data1 = pd.read_excel(data_file_path)
        data1 = pd.read_excel('merge-P-nakłady.xlsx')
        naklady = pd.DataFrame(data1)
        wojewodztwa_naklady = naklady.iloc[47:63]
        wojewodztwa_naklady['wyszczegolnienie'] = wojewodztwa_naklady['wyszczegolnienie'].str.lower()

        lata = [2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022]
        kolumny1 = ['przed_o15', 'przed_o16', 'przed_o17', 'przed_o18', 'przed_o19', 'przed_o20', 'przed_o21', 'przed_o22']
        kolumny2 = ['sprzet_tele15', 'sprzet_tele16', 'sprzet_tele17', 'sprzet_tele18', 'sprzet_tele19', 'sprzet_tele20', 'sprzet_tele21','sprzet_tele22']

        kolumny = [wojewodztwa_naklady[kol2] / wojewodztwa_naklady[kol1] for kol1, kol2 in zip(kolumny1, kolumny2)]
        dane_mapy = {rok: wojewodztwa_naklady[['wyszczegolnienie']].copy().assign(wartosc=kol) for rok, kol in zip(lata, kolumny)}
        
        geojson_url = 'https://raw.githubusercontent.com/ppatrzyk/polska-geojson/master/wojewodztwa/wojewodztwa-min.geojson'
        woj = gpd.read_file(geojson_url).set_index('nazwa')
        
        mapy = {}
        for rok in lata:
            dane_mapy[rok]['wartosc'] = pd.to_numeric(dane_mapy[rok]['wartosc'], errors='coerce')
            dane_mapy[rok]['text'] = dane_mapy[rok].apply(lambda row: f"{row['wyszczegolnienie'].capitalize()}: {row['wartosc']}", axis=1)
            mapy[rok] = px.choropleth(dane_mapy[rok], geojson=woj, locations="wyszczegolnienie", color="wartosc",
                                      color_continuous_scale="greens", projection="mercator")
            mapy[rok].update_geos(fitbounds="locations", visible=False)

        year = st.sidebar.radio("Wybierz rok", lata)
        st.subheader(f"Mapa dla roku {year}")
        st.plotly_chart(mapy[year], use_container_width=True)
    
    except FileNotFoundError:
        st.error("Nie znaleziono pliku Excel. Upewnij się, że podana ścieżka jest poprawna.")
        return
    
    st.title('Analiza trendów')
    @st.cache_data
    def get_local_data():
        try:
            df = pd.read_excel('nak_sprzet_teleinf_wojewodztwa.xlsx', sheet_name=0)
            return df.set_index('Region')
        except Exception as e:
            st.error(f'Błąd wczytywania danych: {e}')
            return pd.DataFrame()

    df = get_local_data()
    if df.empty:
        return
    
    countries = st.multiselect('Wybierz regiony', list(df.index))
    if not countries:
        st.error('Wybierz co najmniej jeden region.')
        return
    
    data = df.loc[countries].T.reset_index().rename(columns={'index': 'year'})
    data = pd.melt(data, id_vars=['year'], var_name='Region', value_name='Wartość')
    data['year'] = pd.to_numeric(data['year'], errors='coerce').dropna().astype(int)

    chart = alt.Chart(data).mark_line().encode(
        x=alt.X('year:O', title='Rok'),
        y=alt.Y('Wartość:Q', title='Wartość'),
        color=alt.Color('Region:N', title='Województwo')
    ).properties(
        title=alt.TitleParams(
            text='Przedsiębiorstwa, które poniosły nakłady na sprzęt telekomunikacyjny',
            anchor='middle', fontSize=12))
    st.altair_chart(chart, use_container_width=True)
    
    st.write("### Wskaźnik nachylenia Sena")
    slope_data = [{'Region': region, 'Slope': linregress(data[data['Region'] == region]['year'], data[data['Region'] == region]['Wartość'])[0]} for region in countries]
    st.write(pd.DataFrame(slope_data))    
    
    st.write("### Test Manna-Kendalla")
    mk_data = []
    for region in countries:
        result = mk.original_test(data[data['Region'] == region]['Wartość'].values)
        mk_data.append({
            'Region': region,
            'Trend': result.trend,
            'p-value': result.p
        })
    st.write(pd.DataFrame(mk_data)) 
    
def maps_demo6():
    st.title("Odsetek przedsiębiorstw, które poniosły nakłady na leasing urządzeń ICT wśród przedsiębiorstw ogółem")

    try:
        # data_file_path = 'C:/Users/Dell/Desktop/merge-P-nakłady.xlsx'
        # data1 = pd.read_excel(data_file_path)
        data1 = pd.read_excel('merge-P-nakłady.xlsx')
        naklady = pd.DataFrame(data1)
        wojewodztwa_naklady = naklady.iloc[47:63]
        wojewodztwa_naklady['wyszczegolnienie'] = wojewodztwa_naklady['wyszczegolnienie'].str.lower()

        lata = [2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022]
        kolumny1 = ['przed_o15', 'przed_o16', 'przed_o17', 'przed_o18', 'przed_o19', 'przed_o20', 'przed_o21', 'przed_o22']
        kolumny2 = ['leasing15', 'leasing16', 'leasing17', 'leasing18', 'leasing19', 'leasing20', 'leasing21','leasing22']
        
        kolumny = [wojewodztwa_naklady[kol2] / wojewodztwa_naklady[kol1] for kol1, kol2 in zip(kolumny1, kolumny2)]
        dane_mapy = {rok: wojewodztwa_naklady[['wyszczegolnienie']].copy().assign(wartosc=kol) for rok, kol in zip(lata, kolumny)}
        
        geojson_url = 'https://raw.githubusercontent.com/ppatrzyk/polska-geojson/master/wojewodztwa/wojewodztwa-min.geojson'
        woj = gpd.read_file(geojson_url).set_index('nazwa')
        
        mapy = {}
        for rok in lata:
            dane_mapy[rok]['wartosc'] = pd.to_numeric(dane_mapy[rok]['wartosc'], errors='coerce')
            dane_mapy[rok]['text'] = dane_mapy[rok].apply(lambda row: f"{row['wyszczegolnienie'].capitalize()}: {row['wartosc']}", axis=1)
            mapy[rok] = px.choropleth(dane_mapy[rok], geojson=woj, locations="wyszczegolnienie", color="wartosc",
                                      color_continuous_scale="greens", projection="mercator")
            mapy[rok].update_geos(fitbounds="locations", visible=False)

        year = st.sidebar.radio("Wybierz rok", lata)
        st.subheader(f"Mapa dla roku {year}")
        st.plotly_chart(mapy[year], use_container_width=True)
    
    except FileNotFoundError:
        st.error("Nie znaleziono pliku Excel. Upewnij się, że podana ścieżka jest poprawna.")
        return
    
    st.title('Analiza trendów')
    @st.cache_data
    def get_local_data():
        try:
            df = pd.read_excel('nak_leasing_wojewodztwa.xlsx', sheet_name=0)
            return df.set_index('Region')
        except Exception as e:
            st.error(f'Błąd wczytywania danych: {e}')
            return pd.DataFrame()

    df = get_local_data()
    if df.empty:
        return
    
    countries = st.multiselect('Wybierz regiony', list(df.index))
    if not countries:
        st.error('Wybierz co najmniej jeden region.')
        return
    
    data = df.loc[countries].T.reset_index().rename(columns={'index': 'year'})
    data = pd.melt(data, id_vars=['year'], var_name='Region', value_name='Wartość')
    data['year'] = pd.to_numeric(data['year'], errors='coerce').dropna().astype(int)

    chart = alt.Chart(data).mark_line().encode(
        x=alt.X('year:O', title='Rok'),
        y=alt.Y('Wartość:Q', title='Wartość'),
        color=alt.Color('Region:N', title='Województwo')
    ).properties(
        title=alt.TitleParams(
            text='Przedsiębiorstwa, które poniosły nakłady na leasing urządzeń ICT',
            anchor='middle', fontSize=12))
    st.altair_chart(chart, use_container_width=True)
    
    st.write("### Wskaźnik nachylenia Sena")
    slope_data = [{'Region': region, 'Slope': linregress(data[data['Region'] == region]['year'], data[data['Region'] == region]['Wartość'])[0]} for region in countries]
    st.write(pd.DataFrame(slope_data))
    
    st.write("### Test Manna-Kendalla")
    mk_data = []
    for region in countries:
        result = mk.original_test(data[data['Region'] == region]['Wartość'].values)
        mk_data.append({
            'Region': region,
            'Trend': result.trend,
            'p-value': result.p
        })
    st.write(pd.DataFrame(mk_data))

def maps_demo7():
    st.title("Odsetek przedsiębiorstw, które poniosły nakłady na oprogramowanie wśród przedsiębiorstw ogółem")

    try:
        # data_file_path = 'C:/Users/Dell/Desktop/merge-P-nakłady.xlsx'
        # data1 = pd.read_excel(data_file_path)
        data1 = pd.read_excel('merge-P-nakłady.xlsx')
        naklady = pd.DataFrame(data1)
        wojewodztwa_naklady = naklady.iloc[47:63]
        wojewodztwa_naklady['wyszczegolnienie'] = wojewodztwa_naklady['wyszczegolnienie'].str.lower()

        lata = [2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022]
        kolumny1 = ['przed_o15', 'przed_o16', 'przed_o17', 'przed_o18', 'przed_o19', 'przed_o20', 'przed_o21', 'przed_o22']
        kolumny2 = ['oprog15', 'oprog16', 'oprog17', 'oprog18', 'oprog19', 'oprog20', 'oprog21','oprog22']
        
        kolumny = [wojewodztwa_naklady[kol2] / wojewodztwa_naklady[kol1] for kol1, kol2 in zip(kolumny1, kolumny2)]
        dane_mapy = {rok: wojewodztwa_naklady[['wyszczegolnienie']].copy().assign(wartosc=kol) for rok, kol in zip(lata, kolumny)}
        
        geojson_url = 'https://raw.githubusercontent.com/ppatrzyk/polska-geojson/master/wojewodztwa/wojewodztwa-min.geojson'
        woj = gpd.read_file(geojson_url).set_index('nazwa')
        
        mapy = {}
        for rok in lata:
            dane_mapy[rok]['wartosc'] = pd.to_numeric(dane_mapy[rok]['wartosc'], errors='coerce')
            dane_mapy[rok]['text'] = dane_mapy[rok].apply(lambda row: f"{row['wyszczegolnienie'].capitalize()}: {row['wartosc']}", axis=1)
            mapy[rok] = px.choropleth(dane_mapy[rok], geojson=woj, locations="wyszczegolnienie", color="wartosc",
                                      color_continuous_scale="greens", projection="mercator")
            mapy[rok].update_geos(fitbounds="locations", visible=False)

        year = st.sidebar.radio("Wybierz rok", lata)
        st.subheader(f"Mapa dla roku {year}")
        st.plotly_chart(mapy[year], use_container_width=True)
    
    except FileNotFoundError:
        st.error("Nie znaleziono pliku Excel. Upewnij się, że podana ścieżka jest poprawna.")
        return
    
    st.title('Analiza trendów')
    @st.cache_data
    def get_local_data():
        try:
            df = pd.read_excel('nak_oprog_wojewodztwa.xlsx', sheet_name=0)
            return df.set_index('Region')
        except Exception as e:
            st.error(f'Błąd wczytywania danych: {e}')
            return pd.DataFrame()

    df = get_local_data()
    if df.empty:
        return
    
    countries = st.multiselect('Wybierz regiony', list(df.index))
    if not countries:
        st.error('Wybierz co najmniej jeden region.')
        return
    
    data = df.loc[countries].T.reset_index().rename(columns={'index': 'year'})
    data = pd.melt(data, id_vars=['year'], var_name='Region', value_name='Wartość')
    data['year'] = pd.to_numeric(data['year'], errors='coerce').dropna().astype(int)

    chart = alt.Chart(data).mark_line().encode(
        x=alt.X('year:O', title='Rok'),
        y=alt.Y('Wartość:Q', title='Wartość'),
        color=alt.Color('Region:N', title='Województwo')
    ).properties(
        title=alt.TitleParams(
            text='Przedsiębiorstwa, które poniosły nakłady na oprogramowanie',
            anchor='middle', fontSize=12))
    st.altair_chart(chart, use_container_width=True)
    
    st.write("### Wskaźnik nachylenia Sena")
    slope_data = [{'Region': region, 'Slope': linregress(data[data['Region'] == region]['year'], data[data['Region'] == region]['Wartość'])[0]} for region in countries]
    st.write(pd.DataFrame(slope_data))
    
    st.write("### Test Manna-Kendalla")
    mk_data = []
    for region in countries:
        result = mk.original_test(data[data['Region'] == region]['Wartość'].values)
        mk_data.append({
            'Region': region,
            'Trend': result.trend,
            'p-value': result.p
        })
    st.write(pd.DataFrame(mk_data))    
    
def maps_demo_pracownicy():
    st.title("Przedsiębiorstwa zatrudniające pracowników według województw")
    try:
        # data_file_path = 'C:/Users/Dell/Desktop/merge-P-pracownicy.xlsx'
        # data1 = pd.read_excel(data_file_path)
        data1 = pd.read_excel('merge-P-pracownicy.xlsx')
        pracownicy = pd.DataFrame(data1)
        wojewodztwa_pracownicy = pracownicy.iloc[47:63]
        wojewodztwa_pracownicy['wyszczegolnienie'] = wojewodztwa_pracownicy['wyszczegolnienie'].str.lower()

        lata = [2015, 2016, 2017, 2018, 2019, 2020, 2021]
        kolumny = ['przed_o15', 'przed_o16', 'przed_o17', 'przed_o18', 'przed_o19', 'przed_o20', 'przed_o21']
        dane_mapy = {}

        for rok, kolumna in zip(lata, kolumny):
            dane_mapy[rok] = wojewodztwa_pracownicy[['wyszczegolnienie', kolumna]].rename(columns={kolumna: 'wartosc'})
            dane_mapy[rok] = dane_mapy[rok].sort_values(by='wartosc', ascending=False)

        geojson_url = 'https://raw.githubusercontent.com/ppatrzyk/polska-geojson/master/wojewodztwa/wojewodztwa-min.geojson'
        woj = gpd.read_file(geojson_url)
        woj = woj.set_index('nazwa')

        mapy = {}
        kolory = {rok: "greens" for rok in lata}

        for rok in lata:
            dane_mapy[rok]['text'] = dane_mapy[rok].apply(lambda row: f"{row['wyszczegolnienie'].capitalize()}: {row['wartosc']}", axis=1)
            mapy[rok] = px.choropleth(
                data_frame=dane_mapy[rok],
                geojson=woj,
                locations="wyszczegolnienie",
                color="wartosc",
                color_continuous_scale=kolory[rok],
                projection="mercator",
                hover_data={"wartosc": True, "wyszczegolnienie": False},
            )
            mapy[rok].update_geos(fitbounds="locations", visible=False)
            mapy[rok].update_traces(text=dane_mapy[rok]['text'], hovertemplate="<b>%{text}</b><extra></extra>")

        year = st.sidebar.radio("Wybierz rok", lata)
        st.subheader(f"Mapa dla roku {year}")
        st.plotly_chart(mapy[year], use_container_width=True)
    except FileNotFoundError:
        st.error("Nie znaleziono pliku Excel. Upewnij się, że podana ścieżka jest poprawna.")
    except Exception as e:
        st.error(f"Wystąpił błąd: {e}")

    st.title('Analiza trendów')
    @st.cache_data
    def get_local_data():
        try:
            df = pd.read_excel('pr_przed_wojewodztwa.xlsx', sheet_name=0)
            return df.set_index('Region')
        except Exception as e:
            st.error(f'Error loading data: {e}')
            return pd.DataFrame()

    df = get_local_data()
    if df.empty:
        return

    countries = st.multiselect('Wybierz regiony', list(df.index))
    if not countries:
        st.error('Wybierz co najmniej jeden region.')
        return

    data = df.loc[countries].T.reset_index().rename(columns={'index': 'year'})
    data = pd.melt(data, id_vars=['year'], var_name='Region', value_name='Wartość')
    data['year'] = pd.to_numeric(data['year'], errors='coerce')
    data = data.dropna(subset=['year']).astype({'year': int})

    chart = alt.Chart(data).mark_line().encode(
        x=alt.X('year:O', title='Rok'),
        y=alt.Y('Wartość:Q', title='Wartość'),
        color=alt.Color('Region:N', title='Województwo')
    ).properties(
        title=alt.TitleParams(
            text='Przedsiębiorstwa zatrudniające pracowników',
            anchor='middle'))
    st.altair_chart(chart, use_container_width=True)
    
    st.write("### Wskaźnik nachylenia Sena")
    slope_data = [{'Region': region, 'Slope': linregress(data[data['Region'] == region]['year'], data[data['Region'] == region]['Wartość'])[0]} for region in countries]
    st.write(pd.DataFrame(slope_data))
    
    st.write("### Test Manna-Kendalla")
    mk_data = []
    for region in countries:
        result = mk.original_test(data[data['Region'] == region]['Wartość'].values)
        mk_data.append({
            'Region': region,
            'Trend': result.trend,
            'p-value': result.p
        })
    st.write(pd.DataFrame(mk_data))
    
def maps_demo2_pracownicy():
    st.title("Odsetek przedsiębiorstw z dostępem do internetu")
    try:
        # data_file_path = 'C:/Users/Dell/Desktop/merge-P-pracownicy.xlsx'
        # data1 = pd.read_excel(data_file_path)
        data1 = pd.read_excel('merge-P-pracownicy.xlsx')
        naklady = pd.DataFrame(data1)
        wojewodztwa_naklady = naklady.iloc[47:63]
        wojewodztwa_naklady['wyszczegolnienie'] = wojewodztwa_naklady['wyszczegolnienie'].str.lower()

        lata = [2015, 2016, 2017, 2018, 2019, 2020, 2021]
        kolumny1 = ['przed_o15', 'przed_o16', 'przed_o17', 'przed_o18', 'przed_o19', 'przed_o20', 'przed_o21']
        kolumny2 = ['przed_di15', 'przed_di16', 'przed_di17', 'przed_di18', 'przed_di19', 'przed_di20', 'przed_di21']
        kolumny = []
        for kol1, kol2 in zip(kolumny1, kolumny2):
            kolumny.append(wojewodztwa_naklady[kol2] / wojewodztwa_naklady[kol1])

        dane_mapy = {}
        for rok, kolumna in zip(lata, kolumny):
            dane_mapy[rok] = wojewodztwa_naklady[['wyszczegolnienie']].copy()
            dane_mapy[rok]['wartosc'] = kolumna
            dane_mapy[rok] = dane_mapy[rok].sort_values(by='wartosc', ascending=False)

        geojson_url = 'https://raw.githubusercontent.com/ppatrzyk/polska-geojson/master/wojewodztwa/wojewodztwa-min.geojson'
        woj = gpd.read_file(geojson_url)
        woj = woj.set_index('nazwa')

        mapy = {}
        kolory = {rok: "greens" for rok in lata}

        for rok in lata:
            dane_mapy[rok]['text'] = dane_mapy[rok].apply(lambda row: f"{row['wyszczegolnienie'].capitalize()}: {row['wartosc']}", axis=1)
            mapy[rok] = px.choropleth(
                data_frame=dane_mapy[rok],
                geojson=woj,
                locations="wyszczegolnienie",
                color="wartosc",
                color_continuous_scale=kolory[rok],
                projection="mercator",
                hover_data={"wartosc": True, "wyszczegolnienie": False},
            )
            mapy[rok].update_geos(fitbounds="locations", visible=False)
            mapy[rok].update_traces(text=dane_mapy[rok]['text'], hovertemplate="<b>%{text}</b><extra></extra>")

        year = st.sidebar.radio("Wybierz rok", lata)
        st.subheader(f"Mapa dla roku {year}")
        st.plotly_chart(mapy[year], use_container_width=True)
    except FileNotFoundError:
        st.error("Nie znaleziono pliku Excel. Upewnij się, że podana ścieżka jest poprawna.")
    except Exception as e:
        st.error(f"Wystąpił błąd: {e}")

    st.title('Analiza trendów')
    @st.cache_data
    def get_local_data():
        try:
            df = pd.read_excel('pred_pol_przed_di_wojewodztwa.xlsx', sheet_name=0)
            return df.set_index('Region')
        except Exception as e:
            st.error(f'Error loading data: {e}')
            return pd.DataFrame()

    df = get_local_data()
    if df.empty:
        return

    countries = st.multiselect('Wybierz regiony', list(df.index))
    if not countries:
        st.error('Wybierz co najmniej jeden region.')
        return

    data = df.loc[countries].T.reset_index().rename(columns={'index': 'year'})
    data = pd.melt(data, id_vars=['year'], var_name='Region', value_name='Wartość')
    data['year'] = pd.to_numeric(data['year'], errors='coerce')
    data = data.dropna(subset=['year']).astype({'year': int})

    chart = alt.Chart(data).mark_line().encode(
        x=alt.X('year:O', title='Rok'),
        y=alt.Y('Wartość:Q', title='Wartość'),
        color=alt.Color('Region:N', title='Województwo')
    ).properties(
        title=alt.TitleParams(
            text='Pracownicy z dostępem do internetu',
            anchor='middle'))
    st.altair_chart(chart, use_container_width=True)
    
    st.write("### Wskaźnik nachylenia Sena")
    slope_data = [{'Region': region, 'Slope': linregress(data[data['Region'] == region]['year'], data[data['Region'] == region]['Wartość'])[0]} for region in countries]
    st.write(pd.DataFrame(slope_data))
         
    st.write("### Test Manna-Kendalla")
    mk_data = []
    for region in countries:
        result = mk.original_test(data[data['Region'] == region]['Wartość'].values)
        mk_data.append({
            'Region': region,
            'Trend': result.trend,
            'p-value': result.p
        })
    st.write(pd.DataFrame(mk_data))
    
def maps_demo3_pracownicy():
    st.title("Pracujący ogółem według województw")
    try:
        # data_file_path = 'C:/Users/Dell/Desktop/merge-P-pracownicy.xlsx'
        # data1 = pd.read_excel(data_file_path)
        data1 = pd.read_excel('merge-P-pracownicy.xlsx')
        pracownicy = pd.DataFrame(data1)
        wojewodztwa_pracownicy = pracownicy.iloc[47:63]
        wojewodztwa_pracownicy['wyszczegolnienie'] = wojewodztwa_pracownicy['wyszczegolnienie'].str.lower()

        lata = [2015, 2016, 2017, 2018, 2019, 2020, 2021]
        kolumny = ['pr_o15', 'pr_o16', 'pr_o17', 'pr_o18', 'pr_o19', 'pr_o20', 'pr_o21','pr_o22']
        dane_mapy = {}

        for rok, kolumna in zip(lata, kolumny):
            dane_mapy[rok] = wojewodztwa_pracownicy[['wyszczegolnienie', kolumna]].rename(columns={kolumna: 'wartosc'})
            dane_mapy[rok] = dane_mapy[rok].sort_values(by='wartosc', ascending=False)

        geojson_url = 'https://raw.githubusercontent.com/ppatrzyk/polska-geojson/master/wojewodztwa/wojewodztwa-min.geojson'
        woj = gpd.read_file(geojson_url)
        woj = woj.set_index('nazwa')

        mapy = {}
        kolory = {rok: "greens" for rok in lata}

        for rok in lata:
            dane_mapy[rok]['text'] = dane_mapy[rok].apply(lambda row: f"{row['wyszczegolnienie'].capitalize()}: {row['wartosc']}", axis=1)
            mapy[rok] = px.choropleth(
                data_frame=dane_mapy[rok],
                geojson=woj,
                locations="wyszczegolnienie",
                color="wartosc",
                color_continuous_scale=kolory[rok],
                projection="mercator",
                hover_data={"wartosc": True, "wyszczegolnienie": False},
            )
            mapy[rok].update_geos(fitbounds="locations", visible=False)
            mapy[rok].update_traces(text=dane_mapy[rok]['text'], hovertemplate="<b>%{text}</b><extra></extra>")

        year = st.sidebar.radio("Wybierz rok", lata)
        st.subheader(f"Mapa dla roku {year}")
        st.plotly_chart(mapy[year], use_container_width=True)
    except FileNotFoundError:
        st.error("Nie znaleziono pliku Excel. Upewnij się, że podana ścieżka jest poprawna.")
    except Exception as e:
        st.error(f"Wystąpił błąd: {e}")

    st.title('Analiza trendów')
    @st.cache_data
    def get_local_data():
        try:
            df = pd.read_excel('pr_o_wojewodztwa.xlsx', sheet_name=0)
            return df.set_index('Region')
        except Exception as e:
            st.error(f'Error loading data: {e}')
            return pd.DataFrame()

    df = get_local_data()
    if df.empty:
        return

    countries = st.multiselect('Wybierz regiony', list(df.index))
    if not countries:
        st.error('Wybierz co najmniej jeden region.')
        return

    data = df.loc[countries].T.reset_index().rename(columns={'index': 'year'})
    data = pd.melt(data, id_vars=['year'], var_name='Region', value_name='Wartość')
    data['year'] = pd.to_numeric(data['year'], errors='coerce')
    data = data.dropna(subset=['year']).astype({'year': int})

    chart = alt.Chart(data).mark_line().encode(
        x=alt.X('year:O', title='Rok'),
        y=alt.Y('Wartość:Q', title='Wartość'),
        color=alt.Color('Region:N', title='Województwo')
    ).properties(
        title=alt.TitleParams(
            text='Pracujący ogółem',
            anchor='middle'))
    st.altair_chart(chart, use_container_width=True)
    
    st.write("### Wskaźnik nachylenia Sena")
    slope_data = [{'Region': region, 'Slope': linregress(data[data['Region'] == region]['year'], data[data['Region'] == region]['Wartość'])[0]} for region in countries]
    st.write(pd.DataFrame(slope_data))
       
    st.write("### Test Manna-Kendalla")
    mk_data = []
    for region in countries:
        result = mk.original_test(data[data['Region'] == region]['Wartość'].values)
        mk_data.append({
            'Region': region,
            'Trend': result.trend,
            'p-value': result.p
        })
    st.write(pd.DataFrame(mk_data))

def maps_demo4_pracownicy():
    st.title("Odsetek pracowników z wyższym wykształceniem")
    try:
        # data_file_path = 'C:/Users/Dell/Desktop/merge-P-pracownicy.xlsx'
        # data1 = pd.read_excel(data_file_path)
        data1 = pd.read_excel('merge-P-pracownicy.xlsx')
        naklady = pd.DataFrame(data1)
        wojewodztwa_naklady = naklady.iloc[47:63]
        wojewodztwa_naklady['wyszczegolnienie'] = wojewodztwa_naklady['wyszczegolnienie'].str.lower()

        lata = [2015, 2016, 2017, 2018, 2019, 2020, 2021]
        kolumny1 = ['przed_o15', 'przed_o16', 'przed_o17', 'przed_o18', 'przed_o19', 'przed_o20', 'przed_o21']
        kolumny2 = ['pr_ww15', 'pr_ww16', 'pr_ww17', 'pr_ww18', 'pr_ww19', 'pr_ww20', 'pr_ww21']

        kolumny = []
        for kol1, kol2 in zip(kolumny1, kolumny2):
            kolumny.append(wojewodztwa_naklady[kol2] / wojewodztwa_naklady[kol1])

        dane_mapy = {}
        for rok, kolumna in zip(lata, kolumny):
            dane_mapy[rok] = wojewodztwa_naklady[['wyszczegolnienie']].copy()
            dane_mapy[rok]['wartosc'] = kolumna
            dane_mapy[rok] = dane_mapy[rok].sort_values(by='wartosc', ascending=False)

        geojson_url = 'https://raw.githubusercontent.com/ppatrzyk/polska-geojson/master/wojewodztwa/wojewodztwa-min.geojson'
        woj = gpd.read_file(geojson_url)
        woj = woj.set_index('nazwa')

        mapy = {}
        kolory = {rok: "greens" for rok in lata}

        for rok in lata:
            dane_mapy[rok]['text'] = dane_mapy[rok].apply(lambda row: f"{row['wyszczegolnienie'].capitalize()}: {row['wartosc']}", axis=1)
            mapy[rok] = px.choropleth(
                data_frame=dane_mapy[rok],
                geojson=woj,
                locations="wyszczegolnienie",
                color="wartosc",
                color_continuous_scale=kolory[rok],
                projection="mercator",
                hover_data={"wartosc": True, "wyszczegolnienie": False},
            )
            mapy[rok].update_geos(fitbounds="locations", visible=False)
            mapy[rok].update_traces(text=dane_mapy[rok]['text'], hovertemplate="<b>%{text}</b><extra></extra>")

        year = st.sidebar.radio("Wybierz rok", lata)
        st.subheader(f"Mapa dla roku {year}")
        st.plotly_chart(mapy[year], use_container_width=True)
    except FileNotFoundError:
        st.error("Nie znaleziono pliku Excel. Upewnij się, że podana ścieżka jest poprawna.")
    except Exception as e:
        st.error(f"Wystąpił błąd: {e}")

    st.title('Analiza trendów')
    @st.cache_data
    def get_local_data():
        try:
            df = pd.read_excel('pr_ww_wojewodztwa.xlsx', sheet_name=0)
            return df.set_index('Region')
        except Exception as e:
            st.error(f'Error loading data: {e}')
            return pd.DataFrame()

    df = get_local_data()
    if df.empty:
        return

    countries = st.multiselect('Wybierz regiony', list(df.index))
    if not countries:
        st.error('Wybierz co najmniej jeden region.')
        return

    data = df.loc[countries].T.reset_index().rename(columns={'index': 'year'})
    data = pd.melt(data, id_vars=['year'], var_name='Region', value_name='Wartość')
    data['year'] = pd.to_numeric(data['year'], errors='coerce')
    data = data.dropna(subset=['year']).astype({'year': int})

    chart = alt.Chart(data).mark_line().encode(
        x=alt.X('year:O', title='Rok'),
        y=alt.Y('Wartość:Q', title='Wartość'),
        color=alt.Color('Region:N', title='Województwo')
    ).properties(
        title=alt.TitleParams(
            text='Pracownicy z wyższym wykształceniem',
            anchor='middle'))
    st.altair_chart(chart, use_container_width=True)
    
    st.write("### Wskaźnik nachylenia Sena")
    slope_data = [{'Region': region, 'Slope': linregress(data[data['Region'] == region]['year'], data[data['Region'] == region]['Wartość'])[0]} for region in countries]
    st.write(pd.DataFrame(slope_data))
        
    st.write("### Test Manna-Kendalla")
    mk_data = []
    for region in countries:
        result = mk.original_test(data[data['Region'] == region]['Wartość'].values)
        mk_data.append({
            'Region': region,
            'Trend': result.trend,
            'p-value': result.p
        })
    st.write(pd.DataFrame(mk_data))
    
def maps_demo5_pracownicy():
    st.title("Odsetek pracowników z dostępem do internetu")
    try:
        # data_file_path = 'C:/Users/Dell/Desktop/merge-P-pracownicy.xlsx'
        # data1 = pd.read_excel(data_file_path)
        data1 = pd.read_excel('merge-P-pracownicy.xlsx')
        naklady = pd.DataFrame(data1)
        wojewodztwa_naklady = naklady.iloc[47:63]
        wojewodztwa_naklady['wyszczegolnienie'] = wojewodztwa_naklady['wyszczegolnienie'].str.lower()

        lata = [2015, 2016, 2017, 2018, 2019, 2020, 2021]
        kolumny1 = ['przed_o15', 'przed_o16', 'przed_o17', 'przed_o18', 'przed_o19', 'przed_o20', 'przed_o21']
        kolumny2 = ['pr_di15', 'pr_di16', 'pr_di17', 'pr_di18', 'pr_di19', 'pr_di20', 'pr_di21']

        kolumny = []
        for kol1, kol2 in zip(kolumny1, kolumny2):
            kolumny.append(wojewodztwa_naklady[kol2] / wojewodztwa_naklady[kol1])

        dane_mapy = {}
        for rok, kolumna in zip(lata, kolumny):
            dane_mapy[rok] = wojewodztwa_naklady[['wyszczegolnienie']].copy()
            dane_mapy[rok]['wartosc'] = kolumna
            dane_mapy[rok] = dane_mapy[rok].sort_values(by='wartosc', ascending=False)
            
        geojson_url = 'https://raw.githubusercontent.com/ppatrzyk/polska-geojson/master/wojewodztwa/wojewodztwa-min.geojson'
        woj = gpd.read_file(geojson_url)
        woj = woj.set_index('nazwa')

        mapy = {}
        kolory = {rok: "greens" for rok in lata}

        for rok in lata:
            dane_mapy[rok]['text'] = dane_mapy[rok].apply(lambda row: f"{row['wyszczegolnienie'].capitalize()}: {row['wartosc']}", axis=1)
            mapy[rok] = px.choropleth(
                data_frame=dane_mapy[rok],
                geojson=woj,
                locations="wyszczegolnienie",
                color="wartosc",
                color_continuous_scale=kolory[rok],
                projection="mercator",
                hover_data={"wartosc": True, "wyszczegolnienie": False},
            )
            mapy[rok].update_geos(fitbounds="locations", visible=False)
            mapy[rok].update_traces(text=dane_mapy[rok]['text'], hovertemplate="<b>%{text}</b><extra></extra>")

        year = st.sidebar.radio("Wybierz rok", lata)
        st.subheader(f"Mapa dla roku {year}")
        st.plotly_chart(mapy[year], use_container_width=True)
    except FileNotFoundError:
        st.error("Nie znaleziono pliku Excel. Upewnij się, że podana ścieżka jest poprawna.")
    except Exception as e:
        st.error(f"Wystąpił błąd: {e}")

    st.title('Analiza trendów')
    @st.cache_data
    def get_local_data():
        try:
            df = pd.read_excel('pr_di_wojewodztwa.xlsx', sheet_name=0)
            return df.set_index('Region')
        except Exception as e:
            st.error(f'Error loading data: {e}')
            return pd.DataFrame()

    df = get_local_data()
    if df.empty:
        return

    countries = st.multiselect('Wybierz regiony', list(df.index))
    if not countries:
        st.error('Wybierz co najmniej jeden region.')
        return

    data = df.loc[countries].T.reset_index().rename(columns={'index': 'year'})
    data = pd.melt(data, id_vars=['year'], var_name='Region', value_name='Wartość')
    data['year'] = pd.to_numeric(data['year'], errors='coerce')
    data = data.dropna(subset=['year']).astype({'year': int})

    chart = alt.Chart(data).mark_line().encode(
        x=alt.X('year:O', title='Rok'),
        y=alt.Y('Wartość:Q', title='Wartość'),
        color=alt.Color('Region:N', title='Województwo')
    ).properties(
        title=alt.TitleParams(
            text='Pracownicy z dostępem do internetu',
            anchor='middle'))
    st.altair_chart(chart, use_container_width=True)
    
    st.write("### Wskaźnik nachylenia Sena")
    slope_data = [{'Region': region, 'Slope': linregress(data[data['Region'] == region]['year'], data[data['Region'] == region]['Wartość'])[0]} for region in countries]
    st.write(pd.DataFrame(slope_data)) 
    
    st.write("### Test Manna-Kendalla")
    mk_data = []
    for region in countries:
        result = mk.original_test(data[data['Region'] == region]['Wartość'].values)
        mk_data.append({
            'Region': region,
            'Trend': result.trend,
            'p-value': result.p
        })
    st.write(pd.DataFrame(mk_data))    
    
def maps_demo6_pracownicy():
    st.title("Odsetek pracowników mających do dyspozycji urządzenia przenośne")
    try:
        # data_file_path = 'C:/Users/Dell/Desktop/merge-P-pracownicy.xlsx'
        # data1 = pd.read_excel(data_file_path)
        data1 = pd.read_excel('merge-P-pracownicy.xlsx')
        naklady = pd.DataFrame(data1)
        wojewodztwa_naklady = naklady.iloc[47:63]
        wojewodztwa_naklady['wyszczegolnienie'] = wojewodztwa_naklady['wyszczegolnienie'].str.lower()

        lata = [2015, 2016, 2017, 2018, 2019, 2020, 2021]
        kolumny1 = ['przed_o15', 'przed_o16', 'przed_o17', 'przed_o18', 'przed_o19', 'przed_o20', 'przed_o21']
        kolumny2 = ['pr_up15', 'pr_up16', 'pr_up17', 'pr_up18', 'pr_up19', 'pr_up20', 'pr_up21']
        kolumny = []
        for kol1, kol2 in zip(kolumny1, kolumny2):
            kolumny.append(wojewodztwa_naklady[kol2] / wojewodztwa_naklady[kol1])

        dane_mapy = {}
        for rok, kolumna in zip(lata, kolumny):
            dane_mapy[rok] = wojewodztwa_naklady[['wyszczegolnienie']].copy()
            dane_mapy[rok]['wartosc'] = kolumna
            dane_mapy[rok] = dane_mapy[rok].sort_values(by='wartosc', ascending=False)
        geojson_url = 'https://raw.githubusercontent.com/ppatrzyk/polska-geojson/master/wojewodztwa/wojewodztwa-min.geojson'
        woj = gpd.read_file(geojson_url)
        woj = woj.set_index('nazwa')

        mapy = {}
        kolory = {rok: "greens" for rok in lata}

        for rok in lata:
            dane_mapy[rok]['text'] = dane_mapy[rok].apply(lambda row: f"{row['wyszczegolnienie'].capitalize()}: {row['wartosc']}", axis=1)
            mapy[rok] = px.choropleth(
                data_frame=dane_mapy[rok],
                geojson=woj,
                locations="wyszczegolnienie",
                color="wartosc",
                color_continuous_scale=kolory[rok],
                projection="mercator",
                hover_data={"wartosc": True, "wyszczegolnienie": False},
            )
            mapy[rok].update_geos(fitbounds="locations", visible=False)
            mapy[rok].update_traces(text=dane_mapy[rok]['text'], hovertemplate="<b>%{text}</b><extra></extra>")

        year = st.sidebar.radio("Wybierz rok", lata)
        st.subheader(f"Mapa dla roku {year}")
        st.plotly_chart(mapy[year], use_container_width=True)
    except FileNotFoundError:
        st.error("Nie znaleziono pliku Excel. Upewnij się, że podana ścieżka jest poprawna.")
    except Exception as e:
        st.error(f"Wystąpił błąd: {e}")

    st.title('Analiza trendów')
    @st.cache_data
    def get_local_data():
        try:
            df = pd.read_excel('pr_up_wojewodztwa.xlsx', sheet_name=0)
            return df.set_index('Region')
        except Exception as e:
            st.error(f'Error loading data: {e}')
            return pd.DataFrame()

    df = get_local_data()
    if df.empty:
        return

    countries = st.multiselect('Wybierz regiony', list(df.index))
    if not countries:
        st.error('Wybierz co najmniej jeden region.')
        return

    data = df.loc[countries].T.reset_index().rename(columns={'index': 'year'})
    data = pd.melt(data, id_vars=['year'], var_name='Region', value_name='Wartość')
    data['year'] = pd.to_numeric(data['year'], errors='coerce')
    data = data.dropna(subset=['year']).astype({'year': int})

    chart = alt.Chart(data).mark_line().encode(
        x=alt.X('year:O', title='Rok'),
        y=alt.Y('Wartość:Q', title='Wartość'),
        color=alt.Color('Region:N', title='Województwo')
    ).properties(
        title=alt.TitleParams(
            text='Pracownicy mający do dyspozycji urządzenia przenośne',
            anchor='middle'))
    st.altair_chart(chart, use_container_width=True)
    
    st.write("### Wskaźnik nachylenia Sena")
    slope_data = [{'Region': region, 'Slope': linregress(data[data['Region'] == region]['year'], data[data['Region'] == region]['Wartość'])[0]} for region in countries]
    st.write(pd.DataFrame(slope_data))
       
    st.write("### Test Manna-Kendalla")
    mk_data = []
    for region in countries:
        result = mk.original_test(data[data['Region'] == region]['Wartość'].values)
        mk_data.append({
            'Region': region,
            'Trend': result.trend,
            'p-value': result.p
        })
    st.write(pd.DataFrame(mk_data))        

def maps_demo_wart_nakladow():
    st.title("Wartość nakładów na sprzęt/oprogramowanie w przedsiębiorstwach według województw")
    try:
        # data_file_path = 'C:/Users/Dell/Desktop/merge-P-wartość_nakładów.xlsx'
        # data1 = pd.read_excel(data_file_path)
        data1 = pd.read_excel('merge-P-wartość_nakładów.xlsx')
        wart_nakladow = pd.DataFrame(data1)
        wojewodztwa_wart_nakladow = wart_nakladow.iloc[47:63]
        wojewodztwa_wart_nakladow['wyszczegolnienie'] = wojewodztwa_wart_nakladow['wyszczegolnienie'].str.lower()

        lata = [2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022]
        kolumny = ['sprzet/opr15', 'sprzet/opr16', 'sprzet/opr17', 'sprzet/opr18', 'sprzet/opr19', 'sprzet/opr20', 'sprzet/opr21', 'sprzet/opr22']
        dane_mapy = {}

        for rok, kolumna in zip(lata, kolumny):
            dane_mapy[rok] = wojewodztwa_wart_nakladow[['wyszczegolnienie', kolumna]].rename(columns={kolumna: 'wartosc'})
            dane_mapy[rok]['wartosc'] = pd.to_numeric(dane_mapy[rok]['wartosc'], errors='coerce')
            dane_mapy[rok] = dane_mapy[rok].dropna(subset=['wartosc'])  
            dane_mapy[rok] = dane_mapy[rok].sort_values(by='wartosc', ascending=False)

        geojson_url = 'https://raw.githubusercontent.com/ppatrzyk/polska-geojson/master/wojewodztwa/wojewodztwa-min.geojson'
        woj = gpd.read_file(geojson_url)
        woj = woj.set_index('nazwa')

        mapy = {}
        kolory = {rok: "greens" for rok in lata}

        for rok in lata:
            dane_mapy[rok]['text'] = dane_mapy[rok].apply(lambda row: f"{row['wyszczegolnienie'].capitalize()}: {row['wartosc']}", axis=1)
            mapy[rok] = px.choropleth(
                data_frame=dane_mapy[rok],
                geojson=woj,
                locations="wyszczegolnienie",
                color="wartosc",
                color_continuous_scale=kolory[rok],
                projection="mercator",
                hover_data={"wartosc": True, "wyszczegolnienie": False},
            )
            mapy[rok].update_geos(fitbounds="locations", visible=False)
            mapy[rok].update_traces(text=dane_mapy[rok]['text'], hovertemplate="<b>%{text}</b><extra></extra>")

        year = st.sidebar.radio("Wybierz rok", lata)
        st.subheader(f"Mapa dla roku {year}")
        st.plotly_chart(mapy[year], use_container_width=True)
    except FileNotFoundError:
        st.error("Nie znaleziono pliku Excel. Upewnij się, że podana ścieżka jest poprawna.")
    except Exception as e:
        st.error(f"Wystąpił błąd: {e}")

    st.title('Analiza trendów')
    @st.cache_data
    def get_local_data():
        try:
            df = pd.read_excel('wart_nak_sprzet_opr_wojewodztwa.xlsx', sheet_name=0)
            return df.set_index('Region')
        except Exception as e:
            st.error(f'Error loading data: {e}')
            return pd.DataFrame()

    df = get_local_data()
    if df.empty:
        return

    countries = st.multiselect('Wybierz regiony', list(df.index))
    if not countries:
        st.error('Wybierz co najmniej jeden region.')
        return

    data = df.loc[countries].T.reset_index().rename(columns={'index': 'year'})
    data = pd.melt(data, id_vars=['year'], var_name='Region', value_name='Wartość')
    data['year'] = pd.to_numeric(data['year'], errors='coerce')
    data = data.dropna(subset=['year']).astype({'year': int})

    chart = alt.Chart(data).mark_line().encode(
        x=alt.X('year:O', title='Rok'),
        y=alt.Y('Wartość:Q', title='Wartość'),
        color=alt.Color('Region:N', title='Województwo')
    ).properties(
        title=alt.TitleParams(
            text='Wartość nakładów na sprzęt i oprogramowanie',
            anchor='middle'))
    st.altair_chart(chart, use_container_width=True)
    
    st.write("### Wskaźnik nachylenia Sena")
    slope_data = [{'Region': region, 'Slope': linregress(data[data['Region'] == region]['year'], data[data['Region'] == region]['Wartość'])[0]} for region in countries]
    st.write(pd.DataFrame(slope_data))
        
    st.write("### Test Manna-Kendalla")
    mk_data = []
    for region in countries:
        result = mk.original_test(data[data['Region'] == region]['Wartość'].values)
        mk_data.append({
            'Region': region,
            'Trend': result.trend,
            'p-value': result.p
        })
    st.write(pd.DataFrame(mk_data))        
        
def maps_demo2_wart_nakladow():
    st.title("Wartość nakładów na sprzęt w przedsiębiorstwach według województw")
    try:
        # data_file_path = 'C:/Users/Dell/Desktop/merge-P-wartość_nakładów.xlsx'
        # data1 = pd.read_excel(data_file_path)
        data1 = pd.read_excel('merge-P-wartość_nakładów.xlsx')
        wart_nakladow = pd.DataFrame(data1)
        wojewodztwa_wart_nakladow = wart_nakladow.iloc[47:63]
        wojewodztwa_wart_nakladow['wyszczegolnienie'] = wojewodztwa_wart_nakladow['wyszczegolnienie'].str.lower()

        lata = [2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022]
        kolumny = ['sprzet15', 'sprzet16', 'sprzet17', 'sprzet18', 'sprzet19', 'sprzet20', 'sprzet21','sprzet22']
        dane_mapy = {}

        for rok, kolumna in zip(lata, kolumny):
            dane_mapy[rok] = wojewodztwa_wart_nakladow[['wyszczegolnienie', kolumna]].rename(columns={kolumna: 'wartosc'})
            dane_mapy[rok]['wartosc'] = pd.to_numeric(dane_mapy[rok]['wartosc'], errors='coerce')
            dane_mapy[rok] = dane_mapy[rok].dropna(subset=['wartosc'])  
            dane_mapy[rok] = dane_mapy[rok].sort_values(by='wartosc', ascending=False)
            
        geojson_url = 'https://raw.githubusercontent.com/ppatrzyk/polska-geojson/master/wojewodztwa/wojewodztwa-min.geojson'
        woj = gpd.read_file(geojson_url)
        woj = woj.set_index('nazwa')

        mapy = {}
        kolory = {rok: "greens" for rok in lata}

        for rok in lata:
            dane_mapy[rok]['text'] = dane_mapy[rok].apply(lambda row: f"{row['wyszczegolnienie'].capitalize()}: {row['wartosc']}", axis=1)
            mapy[rok] = px.choropleth(
                data_frame=dane_mapy[rok],
                geojson=woj,
                locations="wyszczegolnienie",
                color="wartosc",
                color_continuous_scale=kolory[rok],
                projection="mercator",
                hover_data={"wartosc": True, "wyszczegolnienie": False},
            )
            mapy[rok].update_geos(fitbounds="locations", visible=False)
            mapy[rok].update_traces(text=dane_mapy[rok]['text'], hovertemplate="<b>%{text}</b><extra></extra>")

        year = st.sidebar.radio("Wybierz rok", lata)
        st.subheader(f"Mapa dla roku {year}")
        st.plotly_chart(mapy[year], use_container_width=True)
    except FileNotFoundError:
        st.error("Nie znaleziono pliku Excel. Upewnij się, że podana ścieżka jest poprawna.")
    except Exception as e:
        st.error(f"Wystąpił błąd: {e}")

    st.title('Analiza trendów')
    @st.cache_data
    def get_local_data():
        try:
            df = pd.read_excel('wart_nak_sprzet_wojewodztwa.xlsx', sheet_name=0)
            return df.set_index('Region')
        except Exception as e:
            st.error(f'Error loading data: {e}')
            return pd.DataFrame()

    df = get_local_data()
    if df.empty:
        return

    countries = st.multiselect('Wybierz regiony', list(df.index))
    if not countries:
        st.error('Wybierz co najmniej jeden region.')
        return

    data = df.loc[countries].T.reset_index().rename(columns={'index': 'year'})
    data = pd.melt(data, id_vars=['year'], var_name='Region', value_name='Wartość')
    data['year'] = pd.to_numeric(data['year'], errors='coerce')
    data = data.dropna(subset=['year']).astype({'year': int})

    chart = alt.Chart(data).mark_line().encode(
        x=alt.X('year:O', title='Rok'),
        y=alt.Y('Wartość:Q', title='Wartość'),
        color=alt.Color('Region:N', title='Województwo')
    ).properties(
        title=alt.TitleParams(
            text='Wartość nakładów na sprzęt',
            anchor='middle'))
    st.altair_chart(chart, use_container_width=True)
    
    st.write("### Wskaźnik nachylenia Sena")
    slope_data = [{'Region': region, 'Slope': linregress(data[data['Region'] == region]['year'], data[data['Region'] == region]['Wartość'])[0]} for region in countries]
    st.write(pd.DataFrame(slope_data))
          
    st.write("### Test Manna-Kendalla")
    mk_data = []
    for region in countries:
        result = mk.original_test(data[data['Region'] == region]['Wartość'].values)
        mk_data.append({
            'Region': region,
            'Trend': result.trend,
            'p-value': result.p
        })
    st.write(pd.DataFrame(mk_data))   
    
def maps_demo3_wart_nakladow():
    st.title("Nakłady - sprzęt informatyczny")
    try:
        # data_file_path = 'C:/Users/Dell/Desktop/merge-P-wartość_nakładów.xlsx'
        # data1 = pd.read_excel(data_file_path)
        data1 = pd.read_excel('merge-P-wartość_nakładów.xlsx')
        wart_nakladow = pd.DataFrame(data1)
        wojewodztwa_wart_nakladow = wart_nakladow.iloc[47:63]
        wojewodztwa_wart_nakladow['wyszczegolnienie'] = wojewodztwa_wart_nakladow['wyszczegolnienie'].str.lower()

        lata = [2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022]
        kolumny = ['sprzet_inf15', 'sprzet_inf16', 'sprzet_inf17', 'sprzet_inf18', 'sprzet_inf19', 'sprzet_inf20', 'sprzet_inf21','sprzet_inf22']
        dane_mapy = {}

        for rok, kolumna in zip(lata, kolumny):
            dane_mapy[rok] = wojewodztwa_wart_nakladow[['wyszczegolnienie', kolumna]].rename(columns={kolumna: 'wartosc'})
            dane_mapy[rok]['wartosc'] = pd.to_numeric(dane_mapy[rok]['wartosc'], errors='coerce')
            dane_mapy[rok] = dane_mapy[rok].dropna(subset=['wartosc']) 
            dane_mapy[rok] = dane_mapy[rok].sort_values(by='wartosc', ascending=False)
            
        geojson_url = 'https://raw.githubusercontent.com/ppatrzyk/polska-geojson/master/wojewodztwa/wojewodztwa-min.geojson'
        woj = gpd.read_file(geojson_url)
        woj = woj.set_index('nazwa')

        mapy = {}
        kolory = {rok: "greens" for rok in lata}

        for rok in lata:
            dane_mapy[rok]['text'] = dane_mapy[rok].apply(lambda row: f"{row['wyszczegolnienie'].capitalize()}: {row['wartosc']}", axis=1)
            mapy[rok] = px.choropleth(
                data_frame=dane_mapy[rok],
                geojson=woj,
                locations="wyszczegolnienie",
                color="wartosc",
                color_continuous_scale=kolory[rok],
                projection="mercator",
                hover_data={"wartosc": True, "wyszczegolnienie": False},
            )
            mapy[rok].update_geos(fitbounds="locations", visible=False)
            mapy[rok].update_traces(text=dane_mapy[rok]['text'], hovertemplate="<b>%{text}</b><extra></extra>")

        year = st.sidebar.radio("Wybierz rok", lata)
        st.subheader(f"Mapa dla roku {year}")
        st.plotly_chart(mapy[year], use_container_width=True)
    except FileNotFoundError:
        st.error("Nie znaleziono pliku Excel. Upewnij się, że podana ścieżka jest poprawna.")
    except Exception as e:
        st.error(f"Wystąpił błąd: {e}")

    st.title('Analiza trendów')
    @st.cache_data
    def get_local_data():
        try:
            df = pd.read_excel('wart_nak_sprzet_inf_wojewodztwa.xlsx', sheet_name=0)
            return df.set_index('Region')
        except Exception as e:
            st.error(f'Error loading data: {e}')
            return pd.DataFrame()

    df = get_local_data()
    if df.empty:
        return

    countries = st.multiselect('Wybierz regiony', list(df.index))
    if not countries:
        st.error('Wybierz co najmniej jeden region.')
        return

    data = df.loc[countries].T.reset_index().rename(columns={'index': 'year'})
    data = pd.melt(data, id_vars=['year'], var_name='Region', value_name='Wartość')
    data['year'] = pd.to_numeric(data['year'], errors='coerce')
    data = data.dropna(subset=['year']).astype({'year': int})

    chart = alt.Chart(data).mark_line().encode(
        x=alt.X('year:O', title='Rok'),
        y=alt.Y('Wartość:Q', title='Wartość'),
        color=alt.Color('Region:N', title='Województwo')
    ).properties(
        title=alt.TitleParams(
            text='Wartość nakładów na sprzęt informatyczny',
            anchor='middle'))
    st.altair_chart(chart, use_container_width=True)
    
    st.write("### Analiza nachylenia Sen'a")
    slope_data = [{'Region': region, 'Slope': linregress(data[data['Region'] == region]['year'], data[data['Region'] == region]['Wartość'])[0]} for region in countries]
    st.write(pd.DataFrame(slope_data))
    
    st.write("### Test Manna-Kendalla")
    mk_data = []
    for region in countries:
        result = mk.original_test(data[data['Region'] == region]['Wartość'].values)
        mk_data.append({
            'Region': region,
            'Trend': result.trend,
            'p-value': result.p
        })
    st.write(pd.DataFrame(mk_data))
    
def maps_demo4_wart_nakladow():
    st.title("Nakłady - sprzęt telekomunikacyjny")
    try:
        # data_file_path = 'C:/Users/Dell/Desktop/merge-P-wartość_nakładów.xlsx'
        # data1 = pd.read_excel(data_file_path)
        data1 = pd.read_excel('merge-P-wartość_nakładów.xlsx')
        wart_nakladow = pd.DataFrame(data1)
        wojewodztwa_wart_nakladow = wart_nakladow.iloc[47:63]
        wojewodztwa_wart_nakladow['wyszczegolnienie'] = wojewodztwa_wart_nakladow['wyszczegolnienie'].str.lower()

        lata = [2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022]
        kolumny = ['sprzet_tele15', 'sprzet_tele16', 'sprzet_tele17', 'sprzet_tele18', 'sprzet_tele19', 'sprzet_tele20', 'sprzet_tele21','sprzet_tele22']
        dane_mapy = {}

        for rok, kolumna in zip(lata, kolumny):
            dane_mapy[rok] = wojewodztwa_wart_nakladow[['wyszczegolnienie', kolumna]].rename(columns={kolumna: 'wartosc'})
            dane_mapy[rok]['wartosc'] = pd.to_numeric(dane_mapy[rok]['wartosc'], errors='coerce')
            dane_mapy[rok] = dane_mapy[rok].dropna(subset=['wartosc']) 
            dane_mapy[rok] = dane_mapy[rok].sort_values(by='wartosc', ascending=False)
            
        geojson_url = 'https://raw.githubusercontent.com/ppatrzyk/polska-geojson/master/wojewodztwa/wojewodztwa-min.geojson'
        woj = gpd.read_file(geojson_url)
        woj = woj.set_index('nazwa')

        mapy = {}
        kolory = {rok: "greens" for rok in lata}

        for rok in lata:
            dane_mapy[rok]['text'] = dane_mapy[rok].apply(lambda row: f"{row['wyszczegolnienie'].capitalize()}: {row['wartosc']}", axis=1)
            mapy[rok] = px.choropleth(
                data_frame=dane_mapy[rok],
                geojson=woj,
                locations="wyszczegolnienie",
                color="wartosc",
                color_continuous_scale=kolory[rok],
                projection="mercator",
                hover_data={"wartosc": True, "wyszczegolnienie": False},
            )
            mapy[rok].update_geos(fitbounds="locations", visible=False)
            mapy[rok].update_traces(text=dane_mapy[rok]['text'], hovertemplate="<b>%{text}</b><extra></extra>")

        year = st.sidebar.radio("Wybierz rok", lata)
        st.subheader(f"Mapa dla roku {year}")
        st.plotly_chart(mapy[year], use_container_width=True)
    except FileNotFoundError:
        st.error("Nie znaleziono pliku Excel. Upewnij się, że podana ścieżka jest poprawna.")
    except Exception as e:
        st.error(f"Wystąpił błąd: {e}")

    st.title('Analiza trendów')
    @st.cache_data
    def get_local_data():
        try:
            df = pd.read_excel('wart_nak_sprzet_teleinf_wojewodztwa.xlsx', sheet_name=0)
            return df.set_index('Region')
        except Exception as e:
            st.error(f'Error loading data: {e}')
            return pd.DataFrame()

    df = get_local_data()
    if df.empty:
        return

    countries = st.multiselect('Wybierz regiony', list(df.index))
    if not countries:
        st.error('Wybierz co najmniej jeden region.')
        return

    data = df.loc[countries].T.reset_index().rename(columns={'index': 'year'})
    data = pd.melt(data, id_vars=['year'], var_name='Region', value_name='Wartość')
    data['year'] = pd.to_numeric(data['year'], errors='coerce')
    data = data.dropna(subset=['year']).astype({'year': int})

    chart = alt.Chart(data).mark_line().encode(
        x=alt.X('year:O', title='Rok'),
        y=alt.Y('Wartość:Q', title='Wartość'),
        color=alt.Color('Region:N', title='Województwo')
    ).properties(
        title=alt.TitleParams(
            text='Wartość nakładów na sprzęt telekomunikacyjny',
            anchor='middle'))
    st.altair_chart(chart, use_container_width=True)
    
    st.write("### Wskaźnik nachylenia Sena")
    slope_data = [{'Region': region, 'Slope': linregress(data[data['Region'] == region]['year'], data[data['Region'] == region]['Wartość'])[0]} for region in countries]
    st.write(pd.DataFrame(slope_data))
        
    st.write("### Test Manna-Kendalla")
    mk_data = []
    for region in countries:
        result = mk.original_test(data[data['Region'] == region]['Wartość'].values)
        mk_data.append({
            'Region': region,
            'Trend': result.trend,
            'p-value': result.p
        })
    st.write(pd.DataFrame(mk_data))  
    
def maps_demo5_wart_nakladow():
    st.title("Wartość nakładów - leasing")
    try:
        # data_file_path = 'C:/Users/Dell/Desktop/merge-P-wartość_nakładów.xlsx'
        # data1 = pd.read_excel(data_file_path)
        data1 = pd.read_excel('merge-P-wartość_nakładów.xlsx')
        wart_nakladow = pd.DataFrame(data1)
        wojewodztwa_wart_nakladow = wart_nakladow.iloc[47:63]
        wojewodztwa_wart_nakladow['wyszczegolnienie'] = wojewodztwa_wart_nakladow['wyszczegolnienie'].str.lower()

        lata = [2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022]
        kolumny = ['leasing15', 'leasing16', 'leasing17', 'leasing18', 'leasing19', 'leasing20', 'leasing21','leasing22']
        dane_mapy = {}

        for rok, kolumna in zip(lata, kolumny):
            dane_mapy[rok] = wojewodztwa_wart_nakladow[['wyszczegolnienie', kolumna]].rename(columns={kolumna: 'wartosc'})
            dane_mapy[rok]['wartosc'] = pd.to_numeric(dane_mapy[rok]['wartosc'], errors='coerce')
            dane_mapy[rok] = dane_mapy[rok].dropna(subset=['wartosc']) 
            dane_mapy[rok] = dane_mapy[rok].sort_values(by='wartosc', ascending=False)
            
        geojson_url = 'https://raw.githubusercontent.com/ppatrzyk/polska-geojson/master/wojewodztwa/wojewodztwa-min.geojson'
        woj = gpd.read_file(geojson_url)
        woj = woj.set_index('nazwa')

        mapy = {}
        kolory = {rok: "greens" for rok in lata}

        for rok in lata:
            dane_mapy[rok]['text'] = dane_mapy[rok].apply(lambda row: f"{row['wyszczegolnienie'].capitalize()}: {row['wartosc']}", axis=1)
            mapy[rok] = px.choropleth(
                data_frame=dane_mapy[rok],
                geojson=woj,
                locations="wyszczegolnienie",
                color="wartosc",
                color_continuous_scale=kolory[rok],
                projection="mercator",
                hover_data={"wartosc": True, "wyszczegolnienie": False},
            )
            mapy[rok].update_geos(fitbounds="locations", visible=False)
            mapy[rok].update_traces(text=dane_mapy[rok]['text'], hovertemplate="<b>%{text}</b><extra></extra>")

        year = st.sidebar.radio("Wybierz rok", lata)
        st.subheader(f"Mapa dla roku {year}")
        st.plotly_chart(mapy[year], use_container_width=True)
    except FileNotFoundError:
        st.error("Nie znaleziono pliku Excel. Upewnij się, że podana ścieżka jest poprawna.")
    except Exception as e:
        st.error(f"Wystąpił błąd: {e}")

    st.title('Analiza trendów')
    @st.cache_data
    def get_local_data():
        try:
            df = pd.read_excel('wart_nak_leasing_wojewodztwa.xlsx', sheet_name=0)
            return df.set_index('Region')
        except Exception as e:
            st.error(f'Error loading data: {e}')
            return pd.DataFrame()

    df = get_local_data()
    if df.empty:
        return

    countries = st.multiselect('Wybierz regiony', list(df.index))
    if not countries:
        st.error('Wybierz co najmniej jeden region.')
        return

    data = df.loc[countries].T.reset_index().rename(columns={'index': 'year'})
    data = pd.melt(data, id_vars=['year'], var_name='Region', value_name='Wartość')
    data['year'] = pd.to_numeric(data['year'], errors='coerce')
    data = data.dropna(subset=['year']).astype({'year': int})

    chart = alt.Chart(data).mark_line().encode(
        x=alt.X('year:O', title='Rok'),
        y=alt.Y('Wartość:Q', title='Wartość'),
        color=alt.Color('Region:N', title='Województwo')
    ).properties(
        title=alt.TitleParams(
            text='Wartość nakładów na leasing urządzeń ICT',
            anchor='middle'))
    st.altair_chart(chart, use_container_width=True)
    
    st.write("### Wskaźnik nachylenia Sena")
    slope_data = [{'Region': region, 'Slope': linregress(data[data['Region'] == region]['year'], data[data['Region'] == region]['Wartość'])[0]} for region in countries]
    st.write(pd.DataFrame(slope_data))
        
    st.write("### Test Manna-Kendalla")
    mk_data = []
    for region in countries:
        result = mk.original_test(data[data['Region'] == region]['Wartość'].values)
        mk_data.append({
            'Region': region,
            'Trend': result.trend,
            'p-value': result.p
        })
    st.write(pd.DataFrame(mk_data))
    
def maps_demo6_wart_nakladow():
    st.title("Wartość nakładów - oprogramowanie")
    try:
        # data_file_path = 'C:/Users/Dell/Desktop/merge-P-wartość_nakładów.xlsx'
        # data1 = pd.read_excel(data_file_path)
        data1 = pd.read_excel('merge-P-wartość_nakładów.xlsx')
        wart_nakladow = pd.DataFrame(data1)
        wojewodztwa_wart_nakladow = wart_nakladow.iloc[47:63]
        wojewodztwa_wart_nakladow['wyszczegolnienie'] = wojewodztwa_wart_nakladow['wyszczegolnienie'].str.lower()

        lata = [2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022]
        kolumny = ['oprog15', 'oprog16', 'oprog17', 'oprog18', 'oprog19', 'oprog20', 'oprog21','oprog22']
        dane_mapy = {}

        for rok, kolumna in zip(lata, kolumny):
            dane_mapy[rok] = wojewodztwa_wart_nakladow[['wyszczegolnienie', kolumna]].rename(columns={kolumna: 'wartosc'})
            dane_mapy[rok]['wartosc'] = pd.to_numeric(dane_mapy[rok]['wartosc'], errors='coerce')
            dane_mapy[rok] = dane_mapy[rok].dropna(subset=['wartosc']) 
            dane_mapy[rok] = dane_mapy[rok].sort_values(by='wartosc', ascending=False)
            
        geojson_url = 'https://raw.githubusercontent.com/ppatrzyk/polska-geojson/master/wojewodztwa/wojewodztwa-min.geojson'
        woj = gpd.read_file(geojson_url)
        woj = woj.set_index('nazwa')

        mapy = {}
        kolory = {rok: "greens" for rok in lata}

        for rok in lata:
            dane_mapy[rok]['text'] = dane_mapy[rok].apply(lambda row: f"{row['wyszczegolnienie'].capitalize()}: {row['wartosc']}", axis=1)
            mapy[rok] = px.choropleth(
                data_frame=dane_mapy[rok],
                geojson=woj,
                locations="wyszczegolnienie",
                color="wartosc",
                color_continuous_scale=kolory[rok],
                projection="mercator",
                hover_data={"wartosc": True, "wyszczegolnienie": False},
            )
            mapy[rok].update_geos(fitbounds="locations", visible=False)
            mapy[rok].update_traces(text=dane_mapy[rok]['text'], hovertemplate="<b>%{text}</b><extra></extra>")

        year = st.sidebar.radio("Wybierz rok", lata)
        st.subheader(f"Mapa dla roku {year}")
        st.plotly_chart(mapy[year], use_container_width=True)
    except FileNotFoundError:
        st.error("Nie znaleziono pliku Excel. Upewnij się, że podana ścieżka jest poprawna.")
    except Exception as e:
        st.error(f"Wystąpił błąd: {e}")

    st.title('Analiza trendów')
    @st.cache_data
    def get_local_data():
        try:
            df = pd.read_excel('wart_nak_oprog_wojewodztwa.xlsx', sheet_name=0)
            return df.set_index('Region')
        except Exception as e:
            st.error(f'Error loading data: {e}')
            return pd.DataFrame()

    df = get_local_data()
    if df.empty:
        return

    countries = st.multiselect('Wybierz regiony', list(df.index))
    if not countries:
        st.error('Wybierz co najmniej jeden region.')
        return

    data = df.loc[countries].T.reset_index().rename(columns={'index': 'year'})
    data = pd.melt(data, id_vars=['year'], var_name='Region', value_name='Wartość')
    data['year'] = pd.to_numeric(data['year'], errors='coerce')
    data = data.dropna(subset=['year']).astype({'year': int})
    
    chart = alt.Chart(data).mark_line().encode(
        x=alt.X('year:O', title='Rok'),
        y=alt.Y('Wartość:Q', title='Wartość'),
        color=alt.Color('Region:N', title='Województwo')
    ).properties(
        title=alt.TitleParams(
            text='Wartość nakładów na oprogramowanie',
            anchor='middle'))
    st.altair_chart(chart, use_container_width=True)
    
    st.write("### Wskaźnik nachylenia Sena")
    slope_data = [{'Region': region, 'Slope': linregress(data[data['Region'] == region]['year'], data[data['Region'] == region]['Wartość'])[0]} for region in countries]
    st.write(pd.DataFrame(slope_data))
         
    st.write("### Test Manna-Kendalla")
    mk_data = []
    for region in countries:
        result = mk.original_test(data[data['Region'] == region]['Wartość'].values)
        mk_data.append({
            'Region': region,
            'Trend': result.trend,
            'p-value': result.p
        })
    st.write(pd.DataFrame(mk_data)) 
    
def maps_demo2_rodz_polaczen():
    st.title("Odsetek przedsiębiorstw, które mają dostęp do szerokopasmowego łącza internetowego")

    try:
        # data_file_path = 'C:/Users/Dell/Desktop/merge-P-rodzaje_połączeń.xlsx'
        # data1 = pd.read_excel(data_file_path)
        data1 = pd.read_excel('merge-P-rodzaje_połączeń.xlsx')
        naklady = pd.DataFrame(data1)
        wojewodztwa_naklady = naklady.iloc[47:63]
        wojewodztwa_naklady['wyszczegolnienie'] = wojewodztwa_naklady['wyszczegolnienie'].str.lower()

        lata = [2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022]
        kolumny1 = ['przed_o15', 'przed_o16', 'przed_o17', 'przed_o18', 'przed_o19', 'przed_o20', 'przed_o21', 'przed_o22']
        kolumny2 = ['szer_o15', 'szer_o16', 'szer_o17', 'szer_o18', 'szer_o19', 'szer_o20', 'szer_o21','szer_o22']

        kolumny = []
        for kol1, kol2 in zip(kolumny1, kolumny2):
            kolumny.append(wojewodztwa_naklady[kol2] / wojewodztwa_naklady[kol1])

        dane_mapy = {}
        for rok, kolumna in zip(lata, kolumny):
            dane_mapy[rok] = wojewodztwa_naklady[['wyszczegolnienie']].copy()
            dane_mapy[rok]['wartosc'] = kolumna
            dane_mapy[rok] = dane_mapy[rok].sort_values(by='wartosc', ascending=False)
        
        geojson_url = 'https://raw.githubusercontent.com/ppatrzyk/polska-geojson/master/wojewodztwa/wojewodztwa-min.geojson'
        woj = gpd.read_file(geojson_url).set_index('nazwa')
        
        mapy = {}
        for rok in lata:
            dane_mapy[rok]['text'] = dane_mapy[rok].apply(lambda row: f"{row['wyszczegolnienie'].capitalize()}: {row['wartosc']}", axis=1)
            mapy[rok] = px.choropleth(dane_mapy[rok], geojson=woj, locations="wyszczegolnienie", color="wartosc",
                                      color_continuous_scale="greens", projection="mercator")
            mapy[rok].update_geos(fitbounds="locations", visible=False)

        year = st.sidebar.radio("Wybierz rok", lata)
        st.subheader(f"Mapa dla roku {year}")
        st.plotly_chart(mapy[year], use_container_width=True)
    
    except FileNotFoundError:
        st.error("Nie znaleziono pliku Excel. Upewnij się, że podana ścieżka jest poprawna.")
        return
    
    st.title('Analiza trendów')
    @st.cache_data
    def get_local_data():
        try:
            df = pd.read_excel('rodz_pol_szer_wojewodztwa.xlsx', sheet_name=0)
            return df.set_index('Region')
        except Exception as e:
            st.error(f'Błąd wczytywania danych: {e}')
            return pd.DataFrame()

    df = get_local_data()
    if df.empty:
        return
    
    countries = st.multiselect('Wybierz regiony', list(df.index))
    if not countries:
        st.error('Wybierz co najmniej jeden region.')
        return
    
    data = df.loc[countries].T.reset_index().rename(columns={'index': 'year'})
    data = pd.melt(data, id_vars=['year'], var_name='Region', value_name='Wartość')
    data['year'] = pd.to_numeric(data['year'], errors='coerce').dropna().astype(int)

    chart = alt.Chart(data).mark_line().encode(
        x=alt.X('year:O', title='Rok'),
        y=alt.Y('Wartość:Q', title='Wartość'),
        color=alt.Color('Region:N', title='Województwo')
    ).properties(
        title=alt.TitleParams(
            text='Przedsiębiorstwa, które mają dostęp do szerokopasmowego łącza internetowego',
            anchor='middle',fontSize=12))
    st.altair_chart(chart, use_container_width=True)
    
    st.write("### Wskaźnik nachylenia Sena")
    slope_data = [{'Region': region, 'Slope': linregress(data[data['Region'] == region]['year'], data[data['Region'] == region]['Wartość'])[0]} for region in countries]
    st.write(pd.DataFrame(slope_data))
        
    st.write("### Test Manna-Kendalla")
    mk_data = []
    for region in countries:
        result = mk.original_test(data[data['Region'] == region]['Wartość'].values)
        mk_data.append({
            'Region': region,
            'Trend': result.trend,
            'p-value': result.p
        })
    st.write(pd.DataFrame(mk_data)) 
    
def maps_demo3_rodz_polaczen():
    st.title("Odsetek przedsiębiorstw, które posiadają szerokopasmowy  dostęp do Internetu poprzez łącze DSL")

    try:
        # data_file_path = 'C:/Users/Dell/Desktop/merge-P-rodzaje_połączeń.xlsx'
        # data1 = pd.read_excel(data_file_path)
        data1 = pd.read_excel('merge-P-rodzaje_połączeń.xlsx')
        naklady = pd.DataFrame(data1)
        wojewodztwa_naklady = naklady.iloc[47:63]
        wojewodztwa_naklady['wyszczegolnienie'] = wojewodztwa_naklady['wyszczegolnienie'].str.lower()

        lata = [2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022]
        kolumny1 = ['szer_o15', 'szer_o16', 'szer_o17', 'szer_o18', 'szer_o19', 'szer_o20', 'szer_o21','szer_o22']
        kolumny2 = ['szer_DSL15', 'szer_DSL16', 'szer_DSL17', 'szer_DSL18', 'szer_DSL19', 'szer_DSL20', 'szer_DSL21','szer_DSL22']

        kolumny = []
        for kol1, kol2 in zip(kolumny1, kolumny2):
            kolumny.append(wojewodztwa_naklady[kol2] / wojewodztwa_naklady[kol1])

        dane_mapy = {}
        for rok, kolumna in zip(lata, kolumny):
            dane_mapy[rok] = wojewodztwa_naklady[['wyszczegolnienie']].copy()
            dane_mapy[rok]['wartosc'] = kolumna
            dane_mapy[rok] = dane_mapy[rok].sort_values(by='wartosc', ascending=False)

        
        geojson_url = 'https://raw.githubusercontent.com/ppatrzyk/polska-geojson/master/wojewodztwa/wojewodztwa-min.geojson'
        woj = gpd.read_file(geojson_url).set_index('nazwa')
        
        mapy = {}
        for rok in lata:
            dane_mapy[rok]['text'] = dane_mapy[rok].apply(lambda row: f"{row['wyszczegolnienie'].capitalize()}: {row['wartosc']}", axis=1)
            mapy[rok] = px.choropleth(dane_mapy[rok], geojson=woj, locations="wyszczegolnienie", color="wartosc",
                                      color_continuous_scale="greens", projection="mercator")
            mapy[rok].update_geos(fitbounds="locations", visible=False)

        year = st.sidebar.radio("Wybierz rok", lata)
        st.subheader(f"Mapa dla roku {year}")
        st.plotly_chart(mapy[year], use_container_width=True)
    
    except FileNotFoundError:
        st.error("Nie znaleziono pliku Excel. Upewnij się, że podana ścieżka jest poprawna.")
        return
    
    st.title('Analiza trendów')
    @st.cache_data
    def get_local_data():
        try:
            df = pd.read_excel('rodz_pol_szerDSL_wojewodztwa.xlsx', sheet_name=0)
            return df.set_index('Region')
        except Exception as e:
            st.error(f'Błąd wczytywania danych: {e}')
            return pd.DataFrame()

    df = get_local_data()
    if df.empty:
        return
    
    countries = st.multiselect('Wybierz regiony', list(df.index))
    if not countries:
        st.error('Wybierz co najmniej jeden region.')
        return
    
    data = df.loc[countries].T.reset_index().rename(columns={'index': 'year'})
    data = pd.melt(data, id_vars=['year'], var_name='Region', value_name='Wartość')
    data['year'] = pd.to_numeric(data['year'], errors='coerce').dropna().astype(int)

    chart = alt.Chart(data).mark_line().encode(
        x=alt.X('year:O', title='Rok'),
        y=alt.Y('Wartość:Q', title='Wartość'),
        color=alt.Color('Region:N', title='Województwo')
    ).properties(
        title=alt.TitleParams(
            text='Przedsiębiorstwa, które posiadają szerokopasmowy  dostęp do Internetu poprzez łącze DSL',
            anchor='middle',fontSize=12))
    st.altair_chart(chart, use_container_width=True)
    
    st.write("### Wskaźnik nachylenia Sena")
    slope_data = [{'Region': region, 'Slope': linregress(data[data['Region'] == region]['year'], data[data['Region'] == region]['Wartość'])[0]} for region in countries]
    st.write(pd.DataFrame(slope_data))
         
    st.write("### Test Manna-Kendalla")
    mk_data = []
    for region in countries:
        result = mk.original_test(data[data['Region'] == region]['Wartość'].values)
        mk_data.append({
            'Region': region,
            'Trend': result.trend,
            'p-value': result.p
        })
    st.write(pd.DataFrame(mk_data)) 
    
def maps_demo4_pred_polaczen():
    st.title("Odsetek przedsiębiorstw, które mają szerokopasmowy  dostęp do Internetu - prędkość połączenia mniej niż 100 Mbit/s")

    try:
        # data_file_path = 'C:/Users/Dell/Desktop/merge-P-prędkość_połączeń.xlsx'
        # data1 = pd.read_excel(data_file_path)
        data1 = pd.read_excel('merge-P-prędkość_połączeń.xlsx')
        naklady = pd.DataFrame(data1)
        wojewodztwa_naklady = naklady.iloc[47:63]
        wojewodztwa_naklady['wyszczegolnienie'] = wojewodztwa_naklady['wyszczegolnienie'].str.lower()

        lata = [2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022]
        kolumny1 = ['szer_o15', 'szer_o16', 'szer_o17', 'szer_o18', 'szer_o19', 'szer_o20', 'szer_o21','szer_o22']
        kolumny2 = ['szer_100_15', 'szer_100_16', 'szer_100_17', 'szer_100_18', 'szer_100_19', 'szer_100_20', 'szer_100_21','szer_100_22']

        kolumny = []
        for kol1, kol2 in zip(kolumny1, kolumny2):
            kolumny.append(wojewodztwa_naklady[kol2] / wojewodztwa_naklady[kol1])

        dane_mapy = {}
        for rok, kolumna in zip(lata, kolumny):
            dane_mapy[rok] = wojewodztwa_naklady[['wyszczegolnienie']].copy()
            dane_mapy[rok]['wartosc'] = kolumna
            dane_mapy[rok] = dane_mapy[rok].sort_values(by='wartosc', ascending=False)
        
        geojson_url = 'https://raw.githubusercontent.com/ppatrzyk/polska-geojson/master/wojewodztwa/wojewodztwa-min.geojson'
        woj = gpd.read_file(geojson_url).set_index('nazwa')
        
        mapy = {}
        for rok in lata:
            dane_mapy[rok]['text'] = dane_mapy[rok].apply(lambda row: f"{row['wyszczegolnienie'].capitalize()}: {row['wartosc']}", axis=1)
            mapy[rok] = px.choropleth(dane_mapy[rok], geojson=woj, locations="wyszczegolnienie", color="wartosc",
                                      color_continuous_scale="greens", projection="mercator")
            mapy[rok].update_geos(fitbounds="locations", visible=False)

        year = st.sidebar.radio("Wybierz rok", lata)
        st.subheader(f"Mapa dla roku {year}")
        st.plotly_chart(mapy[year], use_container_width=True)
    
    except FileNotFoundError:
        st.error("Nie znaleziono pliku Excel. Upewnij się, że podana ścieżka jest poprawna.")
        return
    
    st.title('Analiza trendów')
    @st.cache_data
    def get_local_data():
        try:
            df = pd.read_excel('pred_pol_szer_w_wojewodztwa.xlsx', sheet_name=0)
            return df.set_index('Region')
        except Exception as e:
            st.error(f'Błąd wczytywania danych: {e}')
            return pd.DataFrame()

    df = get_local_data()
    if df.empty:
        return
    
    countries = st.multiselect('Wybierz regiony', list(df.index))
    if not countries:
        st.error('Wybierz co najmniej jeden region.')
        return
    
    data = df.loc[countries].T.reset_index().rename(columns={'index': 'year'})
    data = pd.melt(data, id_vars=['year'], var_name='Region', value_name='Wartość')
    data['year'] = pd.to_numeric(data['year'], errors='coerce').dropna().astype(int)

    chart = alt.Chart(data).mark_line().encode(
        x=alt.X('year:O', title='Rok'),
        y=alt.Y('Wartość:Q', title='Wartość'),
        color=alt.Color('Region:N', title='Województwo')
    ).properties(
        title=alt.TitleParams(
            text='Przedsiębiorstwa z łączem szerokopasmowym o prędkości <= 100 Mbit/s',
            anchor='middle',fontSize=12))
    st.altair_chart(chart, use_container_width=True)
    
    st.write("### Wskaźnik nachylenia Sena")
    slope_data = [{'Region': region, 'Slope': linregress(data[data['Region'] == region]['year'], data[data['Region'] == region]['Wartość'])[0]} for region in countries]
    st.write(pd.DataFrame(slope_data))
        
    st.write("### Test Manna-Kendalla")
    mk_data = []
    for region in countries:
        result = mk.original_test(data[data['Region'] == region]['Wartość'].values)
        mk_data.append({
            'Region': region,
            'Trend': result.trend,
            'p-value': result.p
        })
    st.write(pd.DataFrame(mk_data)) 
    
def maps_demo5_pred_polaczen():
    st.title("Odsetek przedsiębiorstw, które posiadają szerokopasmowy  dostęp do Internetu - prędkość połączenia przynajmniej 100 Mbit/s")

    try:
        # data_file_path = 'C:/Users/Dell/Desktop/merge-P-prędkość_połączeń.xlsx'
        # data1 = pd.read_excel(data_file_path)
        data1 = pd.read_excel('merge-P-prędkość_połączeń.xlsx')
        naklady = pd.DataFrame(data1)
        wojewodztwa_naklady = naklady.iloc[47:63]
        wojewodztwa_naklady['wyszczegolnienie'] = wojewodztwa_naklady['wyszczegolnienie'].str.lower()

        lata = [2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022]
        kolumny1 = ['szer_o15', 'szer_o16', 'szer_o17', 'szer_o18', 'szer_o19', 'szer_o20', 'szer_o21','szer_o22']
        kolumny2 = ['szer_pow100_15', 'szer_pow100_16', 'szer_pow100_17', 'szer_pow100_18', 'szer_pow100_19', 'szer_pow100_20', 'szer_pow100_21','szer_pow100_22']

        kolumny = []
        for kol1, kol2 in zip(kolumny1, kolumny2):
            kolumny.append(wojewodztwa_naklady[kol2] / wojewodztwa_naklady[kol1])

        dane_mapy = {}
        for rok, kolumna in zip(lata, kolumny):
            dane_mapy[rok] = wojewodztwa_naklady[['wyszczegolnienie']].copy()
            dane_mapy[rok]['wartosc'] = kolumna
            dane_mapy[rok] = dane_mapy[rok].sort_values(by='wartosc', ascending=False)

        geojson_url = 'https://raw.githubusercontent.com/ppatrzyk/polska-geojson/master/wojewodztwa/wojewodztwa-min.geojson'
        woj = gpd.read_file(geojson_url).set_index('nazwa')
        
        mapy = {}
        for rok in lata:
            dane_mapy[rok]['text'] = dane_mapy[rok].apply(lambda row: f"{row['wyszczegolnienie'].capitalize()}: {row['wartosc']}", axis=1)
            mapy[rok] = px.choropleth(dane_mapy[rok], geojson=woj, locations="wyszczegolnienie", color="wartosc",
                                      color_continuous_scale="greens", projection="mercator")
            mapy[rok].update_geos(fitbounds="locations", visible=False)

        year = st.sidebar.radio("Wybierz rok", lata)
        st.subheader(f"Mapa dla roku {year}")
        st.plotly_chart(mapy[year], use_container_width=True)
    
    except FileNotFoundError:
        st.error("Nie znaleziono pliku Excel. Upewnij się, że podana ścieżka jest poprawna.")
        return
    
    st.title('Analiza trendów')
    @st.cache_data
    def get_local_data():
        try:
            df = pd.read_excel('pred_pol_szer_s_wojewodztwa.xlsx', sheet_name=0)
            return df.set_index('Region')
        except Exception as e:
            st.error(f'Błąd wczytywania danych: {e}')
            return pd.DataFrame()

    df = get_local_data()
    if df.empty:
        return
    
    countries = st.multiselect('Wybierz regiony', list(df.index))
    if not countries:
        st.error('Wybierz co najmniej jeden region.')
        return
    
    data = df.loc[countries].T.reset_index().rename(columns={'index': 'year'})
    data = pd.melt(data, id_vars=['year'], var_name='Region', value_name='Wartość')
    data['year'] = pd.to_numeric(data['year'], errors='coerce').dropna().astype(int)

    chart = alt.Chart(data).mark_line().encode(
        x=alt.X('year:O', title='Rok'),
        y=alt.Y('Wartość:Q', title='Wartość'),
        color=alt.Color('Region:N', title='Województwo')
    ).properties(
        title=alt.TitleParams(
            text='Przedsiębiorstwa z łączem szerokopasmowym o prędkości >= 100 Mbit/s',
            anchor='middle',fontSize=12))
    st.altair_chart(chart, use_container_width=True)
    
    st.write("### Wskaźnik nachylenia Sena")
    slope_data = [{'Region': region, 'Slope': linregress(data[data['Region'] == region]['year'], data[data['Region'] == region]['Wartość'])[0]} for region in countries]
    st.write(pd.DataFrame(slope_data))
              
    st.write("### Test Manna-Kendalla")
    mk_data = []
    for region in countries:
        result = mk.original_test(data[data['Region'] == region]['Wartość'].values)
        mk_data.append({
            'Region': region,
            'Trend': result.trend,
            'p-value': result.p
        })
    st.write(pd.DataFrame(mk_data))     
 ########### LICZNOŚĆ FIRM #####################################################################################################
    
def data_visualization():
    selected_subsection = st.sidebar.selectbox("Wybierz kategorię", ["Nakłady", "Pracownicy", "Wartość nakładów", "Rodzaje połączeń", "Prędkość połączeń"])
    if selected_subsection == "Pracownicy":
        pracownicy_section2()
    elif selected_subsection == "Nakłady":
        naklady_section2()
    elif selected_subsection == "Wartość nakładów":
        wartosc_nakladow_section2()
    elif selected_subsection == "Rodzaje połączeń":
        rodzaje_polaczen_section2()
    elif selected_subsection == "Prędkość połączeń":
        predkosc_polaczen_section2()
        
def pracownicy_section2():
    st.sidebar.header("Pracownicy")
    pages = {
        "Przedsiębiorstwa zatrudniające pracowników": data_visualization5,
        "Pracujący ogółem": data_visualization4,
        "Pracownicy z wyższym wykształceniem": data_visualization1,
        "Pracownicy z dostępem do internetu": data_visualization2, 
        "Pracownicy z urządzeniami przenośnymi": data_visualization3,
    }
    selected_page = st.sidebar.selectbox("Wybierz analizę", pages.keys())
    pages[selected_page]()
    
def data_visualization1():
    st.markdown('### Odsetek pracowników z wyższym wykształceniem wśród pracujących ogółem')

    @st.cache_data
    def get_data():
        try:
            df_total = pd.read_csv('transformed_data_pr_o_wielkosc.csv').set_index('Region')
            df_partial = pd.read_csv('transformed_data_pr_ww_wielkosc.csv').set_index('Region')
            return df_total, df_partial
        except Exception as e:
            st.error(f'Error loading data: {e}')
            return pd.DataFrame(), pd.DataFrame()

    df_total, df_partial = get_data()
    if df_total.empty or df_partial.empty:
        return

    countries = st.multiselect('Wybierz wielkość przedsiębiorstw', list(df_partial.index))
    if not countries:
        st.error('Wybierz co najmniej jedną.')
        return

    df_percentage = (df_partial / df_total.replace(0, np.nan)) * 100
    data = df_percentage.loc[countries].T.reset_index().rename(columns={'index': 'year'})
    data = pd.melt(data, id_vars=['year'], var_name='Region', value_name='Procent')
    data['year'] = pd.to_numeric(data['year'], errors='coerce')
    data = data.dropna(subset=['year']).astype({'year': int})

    chart = alt.Chart(data).mark_line().encode(
        x=alt.X('year:O', title='Rok'),
        y=alt.Y('Procent:Q', title='Procent (%)'),
        color=alt.Color('Region:N', title='Wielkość')
    ).properties(
        title=alt.TitleParams(
            text='Odsetek pracowników z wyższym wykształceniem w latach 2015-2022',
            anchor='middle'))
    st.altair_chart(chart, use_container_width=True)

    st.write("### Wskaźnik nachylenia Sena")
    slope_data = []
    for region in countries:
        region_data = data[data['Region'] == region].dropna(subset=['Procent'])
        if len(region_data) > 1:  
            slope, intercept, _, _, _ = linregress(region_data['year'], region_data['Procent'])
            slope_data.append({'Region': region, 'Slope': slope, 'Intercept': intercept})
    sens_slope_results = pd.DataFrame(slope_data)
    st.write(sens_slope_results)

    st.write("### Test Manna-Kendalla")
    mk_data = []
    for region in countries:
        region_data = data[data['Region'] == region].dropna(subset=['Procent'])
        if len(region_data) > 1:
            mk_result = mk.original_test(region_data['Procent'].values)
            mk_data.append({'Region': region, 'Trend': mk_result.trend, 'p-value': mk_result.p})
    mann_kendall_results = pd.DataFrame(mk_data)
    st.write(mann_kendall_results)
       
def data_visualization2():
    st.markdown('### Odsetek pracowników z dostępem do internetu wśród pracujących ogółem')

    @st.cache_data
    def get_data():
        try:
            df_total = pd.read_csv('transformed_data_pr_o_wielkosc.csv').set_index('Region')
            df_partial = pd.read_csv('transformed_data_pr_di_wielkosc.csv').set_index('Region')
            return df_total, df_partial
        except Exception as e:
            st.error(f'Error loading data: {e}')
            return pd.DataFrame(), pd.DataFrame()

    df_total, df_partial = get_data()
    if df_total.empty or df_partial.empty:
        return

    countries = st.multiselect('Wybierz wielkość przedsiębiorstw', list(df_partial.index))
    if not countries:
        st.error('Wybierz co najmniej jedną.')
        return

    df_percentage = (df_partial / df_total.replace(0, np.nan)) * 100
    data = df_percentage.loc[countries].T.reset_index().rename(columns={'index': 'year'})
    data = pd.melt(data, id_vars=['year'], var_name='Region', value_name='Procent')
    data['year'] = pd.to_numeric(data['year'], errors='coerce')
    data = data.dropna(subset=['year']).astype({'year': int})

    chart = alt.Chart(data).mark_line().encode(
        x=alt.X('year:O', title='Rok'),
        y=alt.Y('Procent:Q', title='Procent (%)'),
        color=alt.Color('Region:N', title='Wielkość')
    ).properties(
        title=alt.TitleParams(
            text='Odsetek pracowników z dostępem do internetu w latach 2015-2022',
            anchor='middle'))
    st.altair_chart(chart, use_container_width=True)

    st.write("### Wskaźnik nachylenia Sena")
    slope_data = []
    for region in countries:
        region_data = data[data['Region'] == region].dropna(subset=['Procent'])
        if len(region_data) > 1:  
            slope, intercept, _, _, _ = linregress(region_data['year'], region_data['Procent'])
            slope_data.append({'Region': region, 'Slope': slope, 'Intercept': intercept})
    sens_slope_results = pd.DataFrame(slope_data)
    st.write(sens_slope_results)

    st.write("### Test Manna-Kendalla")
    mk_data = []
    for region in countries:
        region_data = data[data['Region'] == region].dropna(subset=['Procent'])
        if len(region_data) > 1:
            mk_result = mk.original_test(region_data['Procent'].values)
            mk_data.append({'Region': region, 'Trend': mk_result.trend, 'p-value': mk_result.p})
    mann_kendall_results = pd.DataFrame(mk_data)
    st.write(mann_kendall_results)
    
def data_visualization3():
    st.markdown('### Odsetek pracowników mających do dyspozycji urządzenia przenośne wśród pracujących ogółem')

    @st.cache_data
    def get_data():
        try:
            df_total = pd.read_csv('transformed_data_pr_o_wielkosc.csv').set_index('Region')
            df_partial = pd.read_csv('transformed_data_pr_up_wielkosc.csv').set_index('Region')
            return df_total, df_partial
        except Exception as e:
            st.error(f'Error loading data: {e}')
            return pd.DataFrame(), pd.DataFrame()

    df_total, df_partial = get_data()
    if df_total.empty or df_partial.empty:
        return

    countries = st.multiselect('Wybierz wielkość przedsiębiorstw', list(df_partial.index))
    if not countries:
        st.error('Wybierz co najmniej jedną.')
        return

    df_percentage = (df_partial / df_total.replace(0, np.nan)) * 100
    data = df_percentage.loc[countries].T.reset_index().rename(columns={'index': 'year'})
    data = pd.melt(data, id_vars=['year'], var_name='Region', value_name='Procent')
    data['year'] = pd.to_numeric(data['year'], errors='coerce')
    data = data.dropna(subset=['year']).astype({'year': int})
    chart = alt.Chart(data).mark_line().encode(
        x=alt.X('year:O', title='Rok'),
        y=alt.Y('Procent:Q', title='Procent (%)'),
        color=alt.Color('Region:N', title='Wielkość')
    ).properties(
        title=alt.TitleParams(
            text='Odsetek pracowników mających do dyspozycji urządzenia przenośne w latach 2015-2022',
            anchor='middle', fontSize=12))
    st.altair_chart(chart, use_container_width=True)

    st.write("### Wskaźnik nachylenia Sena")
    slope_data = []
    for region in countries:
        region_data = data[data['Region'] == region].dropna(subset=['Procent'])
        if len(region_data) > 1:  
            slope, intercept, _, _, _ = linregress(region_data['year'], region_data['Procent'])
            slope_data.append({'Region': region, 'Slope': slope, 'Intercept': intercept})
    sens_slope_results = pd.DataFrame(slope_data)
    st.write(sens_slope_results)

    st.write("### Test Manna-Kendalla")
    mk_data = []
    for region in countries:
        region_data = data[data['Region'] == region].dropna(subset=['Procent'])
        if len(region_data) > 1:
            mk_result = mk.original_test(region_data['Procent'].values)
            mk_data.append({'Region': region, 'Trend': mk_result.trend, 'p-value': mk_result.p})
    mann_kendall_results = pd.DataFrame(mk_data)
    st.write(mann_kendall_results)
    
def data_visualization4():
    st.markdown('### Pracujący ogółem')

    @st.cache_data
    def get_local_data():
        try:
            df = pd.read_csv('transformed_data_pr_o_wielkosc.csv')
            return df.set_index('Region')
        except Exception as e:
            st.error(f'Error loading data: {e}')
            return pd.DataFrame()

    df = get_local_data()
    if df.empty:
        return

    countries = st.multiselect('Wybierz wielkość przedsiębiorstw', list(df.index))
    if not countries:
        st.error('Wybierz co najmniej jedną.')
        return

    data = df.loc[countries].T.reset_index().rename(columns={'index': 'year'})
    data = pd.melt(data, id_vars=['year'], var_name='Region', value_name='Wartość')
    data['year'] = pd.to_numeric(data['year'], errors='coerce')
    data = data.dropna(subset=['year']).astype({'year': int})

    chart = alt.Chart(data).mark_line().encode(
        x=alt.X('year:O', title='Rok'),
        y=alt.Y('Wartość:Q', title='Wartość'),
        color=alt.Color('Region:N', title='Wielkość')
    ).properties(
        title=alt.TitleParams(
            text='Pracujący ogółem w latach 2015-2022',
            anchor='middle'))
    st.altair_chart(chart, use_container_width=True)

    st.write("### Wskaźnik nachylenia Sena")
    slope_data = []
    for region in countries:
        region_data = data[data['Region'] == region]
        slope, intercept, _, _, _ = linregress(region_data['year'], region_data['Wartość'])
        slope_data.append({'Region': region, 'Slope': slope, 'Intercept': intercept})
    sens_slope_results = pd.DataFrame(slope_data)
    st.write(sens_slope_results)

    st.write("### Test Manna-Kendalla")
    mk_data = []
    for region in countries:
        region_data = data[data['Region'] == region]
        mk_result = mk.original_test(region_data['Wartość'].values)
        mk_data.append({'Region': region, 'Trend': mk_result.trend, 'p-value': mk_result.p})
    mann_kendall_results = pd.DataFrame(mk_data)
    st.write(mann_kendall_results)
       
def data_visualization5():
    st.markdown('### Przedsiębiorstwa zatrudniające pracowników')

    @st.cache_data
    def get_local_data():
        try:
            df = pd.read_csv('transformed_data_przed_o_wielkosc.csv')
            return df.set_index('Region')
        except Exception as e:
            st.error(f'Error loading data: {e}')
            return pd.DataFrame()

    df = get_local_data()
    if df.empty:
        return

    countries = st.multiselect('Wybierz wielkość przedsiębiorstw', list(df.index))
    if not countries:
        st.error('Wybierz co najmniej jeden region.')
        return

    data = df.loc[countries].T.reset_index().rename(columns={'index': 'year'})
    data = pd.melt(data, id_vars=['year'], var_name='Region', value_name='Wartość')
    data['year'] = pd.to_numeric(data['year'], errors='coerce')
    data = data.dropna(subset=['year']).astype({'year': int})

    chart = alt.Chart(data).mark_line().encode(
        x=alt.X('year:O', title='Rok'),
        y=alt.Y('Wartość:Q', title='Wartość'),
        color=alt.Color('Region:N', title='Wielkość')
    ).properties(
        title=alt.TitleParams(
            text='Ilość przedsiębiorstw w latach 2015-2021',
            anchor='middle'))
    st.altair_chart(chart, use_container_width=True)
    
    st.write("### Wskaźnik nachylenia Sena")
    slope_data = []
    for region in countries:
        region_data = data[data['Region'] == region]
        slope, intercept, _, _, _ = linregress(region_data['year'], region_data['Wartość'])
        slope_data.append({'Region': region, 'Slope': slope, 'Intercept': intercept})
    sens_slope_results = pd.DataFrame(slope_data)
    st.write(sens_slope_results)

    st.write("### Test Manna-Kendalla")
    mk_data = []
    for region in countries:
        region_data = data[data['Region'] == region]
        mk_result = mk.original_test(region_data['Wartość'].values)
        mk_data.append({'Region': region, 'Trend': mk_result.trend, 'p-value': mk_result.p})
    mann_kendall_results = pd.DataFrame(mk_data)
    st.write(mann_kendall_results)
        
def naklady_section2():
    st.sidebar.header("Nakłady")
    pages = {
        "Nakłady na sprzęt/oprogramowanie": data_visualization_naklady2,
        "Nakłady - sprzęt ogółem": data_visualization_naklady4,
        "Nakłady - sprzęt informatyczny": data_visualization_naklady3,
        "Nakłady - sprzęt telekomunikacyjny": data_visualization_naklady5,
        "Nakłady - leasing": data_visualization_naklady6,
        "Nakłady - oprogramowanie": data_visualization_naklady7,
    }
    selected_page = st.sidebar.selectbox("Wybierz analizę", pages.keys())
    pages[selected_page]()
    
def data_visualization_naklady2():
    st.markdown('### Odsetek przedsiębiorstw, które poniosły nakłady na sprzęt i oprogramowanie')

    @st.cache_data
    def get_data():
        try:
            df_total = pd.read_csv('transformed_data_naklady_przed_wielkosc.csv').set_index('Region')
            df_partial = pd.read_csv('transformed_data_naklady_sprzetopr_wielkosc.csv').set_index('Region')
            return df_total, df_partial
        except Exception as e:
            st.error(f'Error loading data: {e}')
            return pd.DataFrame(), pd.DataFrame()

    df_total, df_partial = get_data()
    if df_total.empty or df_partial.empty:
        return

    countries = st.multiselect('Wybierz wielkość przedsiębiorstw', list(df_partial.index))
    if not countries:
        st.error('Wybierz co najmniej jedną.')
        return

    df_percentage = (df_partial / df_total.replace(0, np.nan)) * 100
    data = df_percentage.loc[countries].T.reset_index().rename(columns={'index': 'year'})
    data = pd.melt(data, id_vars=['year'], var_name='Region', value_name='Procent')
    data['year'] = pd.to_numeric(data['year'], errors='coerce')
    data = data.dropna(subset=['year']).astype({'year': int})

#     chart = alt.Chart(data).mark_line().encode(
#         x=alt.X('year:O',title='Rok'),y=alt.Y('Procent:Q', title='Procent (%)'),color=alt.Color('Region:N',title='Wielkość'))
    chart = alt.Chart(data).mark_line().encode(
        x=alt.X('year:O', title='Rok'),
        y=alt.Y('Procent:Q', title='Procent (%)'),
        color=alt.Color('Region:N', title='Wielkość')
    ).properties(
        title=alt.TitleParams(
            text='Odsetek przedsiębiorstw, które poniosły nakłady na sprzęt i oprogramowanie w latach 2015-2022',
            anchor='middle', fontSize=12))
    st.altair_chart(chart, use_container_width=True)

    st.write("### Wskaźnik nachylenia Sena")
    slope_data = []
    for region in countries:
        region_data = data[data['Region'] == region].dropna(subset=['Procent'])
        if len(region_data) > 1:  # Sprawdzamy, czy jest wystarczająco dużo danych do regresji
            slope, intercept, _, _, _ = linregress(region_data['year'], region_data['Procent'])
            slope_data.append({'Region': region, 'Slope': slope, 'Intercept': intercept})
    sens_slope_results = pd.DataFrame(slope_data)
    st.write(sens_slope_results)

    st.write("### Test Manna-Kendalla")
    mk_data = []
    for region in countries:
        region_data = data[data['Region'] == region].dropna(subset=['Procent'])
        if len(region_data) > 1:
            mk_result = mk.original_test(region_data['Procent'].values)
            mk_data.append({'Region': region, 'Trend': mk_result.trend, 'p-value': mk_result.p})
    mann_kendall_results = pd.DataFrame(mk_data)
    st.write(mann_kendall_results)

def data_visualization_naklady3():
    st.markdown('### Odsetek przedsiębiorstw, które poniosły nakłady na sprzęt informatyczny')

    @st.cache_data
    def get_data():
        try:
            df_total = pd.read_csv('transformed_data_naklady_przed_wielkosc.csv').set_index('Region')
            df_partial = pd.read_csv('transformed_data_naklady_sprzet_inf_wielkosc.csv') .set_index('Region')
            return df_total, df_partial
        except Exception as e:
            st.error(f'Error loading data: {e}')
            return pd.DataFrame(), pd.DataFrame()

    df_total, df_partial = get_data()
    if df_total.empty or df_partial.empty:
        return

    countries = st.multiselect('Wybierz wielkość przedsiębiorstw', list(df_partial.index))
    if not countries:
        st.error('Wybierz co najmniej jedną.')
        return

    df_percentage = (df_partial / df_total.replace(0, np.nan)) * 100
    data = df_percentage.loc[countries].T.reset_index().rename(columns={'index': 'year'})
    data = pd.melt(data, id_vars=['year'], var_name='Region', value_name='Procent')
    data['year'] = pd.to_numeric(data['year'], errors='coerce')
    data = data.dropna(subset=['year']).astype({'year': int})

#     chart = alt.Chart(data).mark_line().encode(
#         x=alt.X('year:O',title='Rok'),y=alt.Y('Procent:Q', title='Procent (%)'),color=alt.Color('Region:N',title='Wielkość'))
    chart = alt.Chart(data).mark_line().encode(
        x=alt.X('year:O', title='Rok'),
        y=alt.Y('Procent:Q', title='Procent (%)'),
        color=alt.Color('Region:N', title='Wielkość')
    ).properties(
        title=alt.TitleParams(
            text='Odsetek przedsiębiorstw, które poniosły nakłady na sprzęt informatyczny w latach 2015-2022',
            anchor='middle', fontSize=12))
    st.altair_chart(chart, use_container_width=True)

    st.write("### Wskaźnik nachylenia Sena")
    slope_data = []
    for region in countries:
        region_data = data[data['Region'] == region].dropna(subset=['Procent'])
        if len(region_data) > 1:  # Sprawdzamy, czy jest wystarczająco dużo danych do regresji
            slope, intercept, _, _, _ = linregress(region_data['year'], region_data['Procent'])
            slope_data.append({'Region': region, 'Slope': slope, 'Intercept': intercept})
    sens_slope_results = pd.DataFrame(slope_data)
    st.write(sens_slope_results)

    st.write("### Test Manna-Kendalla")
    mk_data = []
    for region in countries:
        region_data = data[data['Region'] == region].dropna(subset=['Procent'])
        if len(region_data) > 1:
            mk_result = mk.original_test(region_data['Procent'].values)
            mk_data.append({'Region': region, 'Trend': mk_result.trend, 'p-value': mk_result.p})
    mann_kendall_results = pd.DataFrame(mk_data)
    st.write(mann_kendall_results)
            
def data_visualization_naklady4():
    st.markdown('### Odsetek przedsiębiorstw, które poniosły nakłady na sprzęt')

    @st.cache_data
    def get_data():
        try:
            df_total = pd.read_csv('transformed_data_naklady_przed_wielkosc.csv').set_index('Region')
            df_partial = pd.read_csv('transformed_data_naklady_sprzet_wielkosc.csv') .set_index('Region')
            return df_total, df_partial
        except Exception as e:
            st.error(f'Error loading data: {e}')
            return pd.DataFrame(), pd.DataFrame()

    df_total, df_partial = get_data()
    if df_total.empty or df_partial.empty:
        return

    countries = st.multiselect('Wybierz wielkość przedsiębiorstw', list(df_partial.index))
    if not countries:
        st.error('Wybierz co najmniej jedną.')
        return

    df_percentage = (df_partial / df_total.replace(0, np.nan)) * 100
    data = df_percentage.loc[countries].T.reset_index().rename(columns={'index': 'year'})
    data = pd.melt(data, id_vars=['year'], var_name='Region', value_name='Procent')
    data['year'] = pd.to_numeric(data['year'], errors='coerce')
    data = data.dropna(subset=['year']).astype({'year': int})

#     chart = alt.Chart(data).mark_line().encode(
#         x=alt.X('year:O',title='Rok'),y=alt.Y('Procent:Q', title='Procent (%)'),color=alt.Color('Region:N',title='Wielkość'))
    chart = alt.Chart(data).mark_line().encode(
        x=alt.X('year:O', title='Rok'),
        y=alt.Y('Procent:Q', title='Procent (%)'),
        color=alt.Color('Region:N', title='Wielkość')
    ).properties(
        title=alt.TitleParams(
            text='Odsetek przedsiębiorstw, które poniosły nakłady na sprzęt w latach 2015-2022',
            anchor='middle'))
    st.altair_chart(chart, use_container_width=True)

    st.write("### Wskaźnik nachylenia Sena")
    slope_data = []
    for region in countries:
        region_data = data[data['Region'] == region].dropna(subset=['Procent'])
        if len(region_data) > 1:  # Sprawdzamy, czy jest wystarczająco dużo danych do regresji
            slope, intercept, _, _, _ = linregress(region_data['year'], region_data['Procent'])
            slope_data.append({'Region': region, 'Slope': slope, 'Intercept': intercept})
    sens_slope_results = pd.DataFrame(slope_data)
    st.write(sens_slope_results)

    st.write("### Test Manna-Kendalla")
    mk_data = []
    for region in countries:
        region_data = data[data['Region'] == region].dropna(subset=['Procent'])
        if len(region_data) > 1:
            mk_result = mk.original_test(region_data['Procent'].values)
            mk_data.append({'Region': region, 'Trend': mk_result.trend, 'p-value': mk_result.p})
    mann_kendall_results = pd.DataFrame(mk_data)
    st.write(mann_kendall_results)
    
def data_visualization_naklady5():
    st.markdown('### Odsetek przedsiębiorstw, które poniosły nakłady na sprzęt telekomunikacyjny')

    @st.cache_data
    def get_data():
        try:
            df_total = pd.read_csv('transformed_data_naklady_przed_wielkosc.csv').set_index('Region')
            df_partial = pd.read_csv('transformed_data_naklady_sprzet_teleinf_wielkosc.csv').set_index('Region')
            return df_total, df_partial
        except Exception as e:
            st.error(f'Error loading data: {e}')
            return pd.DataFrame(), pd.DataFrame()

    df_total, df_partial = get_data()
    if df_total.empty or df_partial.empty:
        return

    countries = st.multiselect('Wybierz wielkość przedsiębiorstw', list(df_partial.index))
    if not countries:
        st.error('Wybierz co najmniej jedną.')
        return

    df_percentage = (df_partial / df_total.replace(0, np.nan)) * 100
    data = df_percentage.loc[countries].T.reset_index().rename(columns={'index': 'year'})
    data = pd.melt(data, id_vars=['year'], var_name='Region', value_name='Procent')
    data['year'] = pd.to_numeric(data['year'], errors='coerce')
    data = data.dropna(subset=['year']).astype({'year': int})

#     chart = alt.Chart(data).mark_line().encode(
#         x=alt.X('year:O',title='Rok'),y=alt.Y('Procent:Q', title='Procent (%)'),color=alt.Color('Region:N',title='Wielkość'))
    chart = alt.Chart(data).mark_line().encode(
        x=alt.X('year:O', title='Rok'),
        y=alt.Y('Procent:Q', title='Procent (%)'),
        color=alt.Color('Region:N', title='Wielkość')
    ).properties(
        title=alt.TitleParams(
            text='Odsetek przedsiębiorstw, które poniosły nakłady na sprzęt telekomunikacyjny w latach 2015-2022',
            anchor='middle', fontSize=12))
    st.altair_chart(chart, use_container_width=True)

    st.write("### Wskaźnik nachylenia Sena")
    slope_data = []
    for region in countries:
        region_data = data[data['Region'] == region].dropna(subset=['Procent'])
        if len(region_data) > 1:  # Sprawdzamy, czy jest wystarczająco dużo danych do regresji
            slope, intercept, _, _, _ = linregress(region_data['year'], region_data['Procent'])
            slope_data.append({'Region': region, 'Slope': slope, 'Intercept': intercept})
    sens_slope_results = pd.DataFrame(slope_data)
    st.write(sens_slope_results)

    st.write("### Test Manna-Kendalla")
    mk_data = []
    for region in countries:
        region_data = data[data['Region'] == region].dropna(subset=['Procent'])
        if len(region_data) > 1:
            mk_result = mk.original_test(region_data['Procent'].values)
            mk_data.append({'Region': region, 'Trend': mk_result.trend, 'p-value': mk_result.p})
    mann_kendall_results = pd.DataFrame(mk_data)
    st.write(mann_kendall_results)    
    
def data_visualization_naklady6():
    st.markdown('### Odsetek przedsiębiorstw, które poniosły nakłady na leasing finansowy urządzeń ICT')

    @st.cache_data
    def get_data():
        try:
            df_total = pd.read_csv('transformed_data_naklady_przed_wielkosc.csv').set_index('Region')
            df_partial = pd.read_csv('transformed_data_naklady_leasing_wielkosc.csv').set_index('Region')
            return df_total, df_partial
        except Exception as e:
            st.error(f'Error loading data: {e}')
            return pd.DataFrame(), pd.DataFrame()

    df_total, df_partial = get_data()
    if df_total.empty or df_partial.empty:
        return

    countries = st.multiselect('Wybierz wielkość przedsiębiorstw', list(df_partial.index))
    if not countries:
        st.error('Wybierz co najmniej jedną.')
        return

    df_percentage = (df_partial / df_total.replace(0, np.nan)) * 100
    data = df_percentage.loc[countries].T.reset_index().rename(columns={'index': 'year'})
    data = pd.melt(data, id_vars=['year'], var_name='Region', value_name='Procent')
    data['year'] = pd.to_numeric(data['year'], errors='coerce')
    data = data.dropna(subset=['year']).astype({'year': int})

#     chart = alt.Chart(data).mark_line().encode(
#         x=alt.X('year:O',title='Rok'),y=alt.Y('Procent:Q', title='Procent (%)'),color=alt.Color('Region:N',title='Wielkość'))
    chart = alt.Chart(data).mark_line().encode(
        x=alt.X('year:O', title='Rok'),
        y=alt.Y('Procent:Q', title='Procent (%)'),
        color=alt.Color('Region:N', title='Wielkość')
    ).properties(
        title=alt.TitleParams(
            text='Odsetek przedsiębiorstw, które poniosły nakłady na leasing finansowy urządzeń ICT w latach 2015-2022',
            anchor='middle', fontSize=12))
    st.altair_chart(chart, use_container_width=True)

    st.write("### Wskaźnik nachylenia Sena")
    slope_data = []
    for region in countries:
        region_data = data[data['Region'] == region].dropna(subset=['Procent'])
        if len(region_data) > 1:  # Sprawdzamy, czy jest wystarczająco dużo danych do regresji
            slope, intercept, _, _, _ = linregress(region_data['year'], region_data['Procent'])
            slope_data.append({'Region': region, 'Slope': slope, 'Intercept': intercept})
    sens_slope_results = pd.DataFrame(slope_data)
    st.write(sens_slope_results)

    st.write("### Test Manna-Kendalla")
    mk_data = []
    for region in countries:
        region_data = data[data['Region'] == region].dropna(subset=['Procent'])
        if len(region_data) > 1:
            mk_result = mk.original_test(region_data['Procent'].values)
            mk_data.append({'Region': region, 'Trend': mk_result.trend, 'p-value': mk_result.p})
    mann_kendall_results = pd.DataFrame(mk_data)
    st.write(mann_kendall_results)  
        
def data_visualization_naklady7():
    st.markdown('### Odsetek przedsiębiorstw, które poniosły nakłady na oprogramowanie')

    @st.cache_data
    def get_data():
        try:
            df_total = pd.read_csv('transformed_data_naklady_przed_wielkosc.csv').set_index('Region')
            df_partial = pd.read_csv('transformed_data_naklady_oprog_wielkosc.csv').set_index('Region')
            return df_total, df_partial
        except Exception as e:
            st.error(f'Error loading data: {e}')
            return pd.DataFrame(), pd.DataFrame()

    df_total, df_partial = get_data()
    if df_total.empty or df_partial.empty:
        return

    countries = st.multiselect('Wybierz wielkość przedsiębiorstw', list(df_partial.index))
    if not countries:
        st.error('Wybierz co najmniej jedną.')
        return

    df_percentage = (df_partial / df_total.replace(0, np.nan)) * 100
    data = df_percentage.loc[countries].T.reset_index().rename(columns={'index': 'year'})
    data = pd.melt(data, id_vars=['year'], var_name='Region', value_name='Procent')
    data['year'] = pd.to_numeric(data['year'], errors='coerce')
    data = data.dropna(subset=['year']).astype({'year': int})

#     chart = alt.Chart(data).mark_line().encode(
#         x=alt.X('year:O',title='Rok'),y=alt.Y('Procent:Q', title='Procent (%)'),color=alt.Color('Region:N',title='Wielkość'))
    chart = alt.Chart(data).mark_line().encode(
        x=alt.X('year:O', title='Rok'),
        y=alt.Y('Procent:Q', title='Procent (%)'),
        color=alt.Color('Region:N', title='Wielkość')
    ).properties(
        title=alt.TitleParams(
            text='Odsetek przedsiębiorstw, które poniosły nakłady na oprogramowanie w latach 2015-2022',
            anchor='middle', fontSize=12))
    st.altair_chart(chart, use_container_width=True)

    st.write("### Wskaźnik nachylenia Sena")
    slope_data = []
    for region in countries:
        region_data = data[data['Region'] == region].dropna(subset=['Procent'])
        if len(region_data) > 1:  # Sprawdzamy, czy jest wystarczająco dużo danych do regresji
            slope, intercept, _, _, _ = linregress(region_data['year'], region_data['Procent'])
            slope_data.append({'Region': region, 'Slope': slope, 'Intercept': intercept})
    sens_slope_results = pd.DataFrame(slope_data)
    st.write(sens_slope_results)

    st.write("### Test Manna-Kendalla")
    mk_data = []
    for region in countries:
        region_data = data[data['Region'] == region].dropna(subset=['Procent'])
        if len(region_data) > 1:
            mk_result = mk.original_test(region_data['Procent'].values)
            mk_data.append({'Region': region, 'Trend': mk_result.trend, 'p-value': mk_result.p})
    mann_kendall_results = pd.DataFrame(mk_data)
    st.write(mann_kendall_results)  
        
def wartosc_nakladow_section2():
    st.sidebar.header("Wartość Nakładów")
    pages = {
        "Wartość nakładów - sprzęt/oprogramowanie": data_visualization_wart_nakladow,
        "Wartość nakładów - sprzęt": data_visualization_wart_nakladow2,
        "Wartość nakładów - sprzęt informatyczny": data_visualization_wart_nakladow3,
        "Wartość nakładów - sprzęt telekomunikacyjny": data_visualization_wart_nakladow4,
        "Wartość nakładów - leasing": data_visualization_wart_nakladow5,
        "Wartość nakładów - oprogramowanie": data_visualization_wart_nakladow6,
    }
    selected_page = st.sidebar.selectbox("Wybierz analizę", pages.keys())
    pages[selected_page]()

def data_visualization_wart_nakladow():
    st.markdown('### Wartość nakładów na sprzęt i oprogramowanie')

    @st.cache_data
    def get_local_data():
        try:
            df = pd.read_csv('transformed_data_wart_nakladow_spr_oprog_wielkosc.csv')
            return df.set_index('Region')
        except Exception as e:
            st.error(f'Error loading data: {e}')
            return pd.DataFrame()

    df = get_local_data()
    if df.empty:
        return

    countries = st.multiselect('Wybierz wielkość przedsiębiorstw', list(df.index))
    if not countries:
        st.error('Wybierz co najmniej jedną.')
        return

    data = df.loc[countries].T.reset_index().rename(columns={'index': 'year'})
    data = pd.melt(data, id_vars=['year'], var_name='Region', value_name='Wartość')
    data['year'] = pd.to_numeric(data['year'], errors='coerce')
    data = data.dropna(subset=['year']).astype({'year': int})

#     chart = alt.Chart(data).mark_line().encode(
#         x=alt.X('year:O',title='Rok'),y=alt.Y('Wartość:Q', title='Wartość'),color=alt.Color('Region:N',title='Wielkość'))
    chart = alt.Chart(data).mark_line().encode(
        x=alt.X('year:O', title='Rok'),
        y=alt.Y('Wartość:Q', title='Wartość'),
        color=alt.Color('Region:N', title='Wielkość')
    ).properties(
        title=alt.TitleParams(
            text='Wartość nakładów na sprzęt i oprogramowanie w latach 2015-2022',
            anchor='middle'))
    
    st.altair_chart(chart, use_container_width=True)

    st.write("### Wskaźnik nachylenia Sena")
    slope_data = []
    for region in countries:
        region_data = data[data['Region'] == region]
        slope, intercept, _, _, _ = linregress(region_data['year'], region_data['Wartość'])
        slope_data.append({'Region': region, 'Slope': slope, 'Intercept': intercept})
    sens_slope_results = pd.DataFrame(slope_data)
    st.write(sens_slope_results)

    st.write("### Test Manna-Kendalla")
    mk_data = []
    for region in countries:
        region_data = data[data['Region'] == region]
        mk_result = mk.original_test(region_data['Wartość'].values)
        mk_data.append({'Region': region, 'Trend': mk_result.trend, 'p-value': mk_result.p})
    mann_kendall_results = pd.DataFrame(mk_data)
    st.write(mann_kendall_results)
    
def data_visualization_wart_nakladow2():
    st.markdown('### Wartość nakładów na sprzęt')

    @st.cache_data
    def get_local_data():
        try:
            df = pd.read_csv('transformed_data_wart_nakladow_sprzet_wielkosc.csv')
            return df.set_index('Region')
        except Exception as e:
            st.error(f'Error loading data: {e}')
            return pd.DataFrame()

    df = get_local_data()
    if df.empty:
        return

    countries = st.multiselect('Wybierz wielkość przedsiębiorstw', list(df.index))
    if not countries:
        st.error('Wybierz co najmniej jedną.')
        return

    data = df.loc[countries].T.reset_index().rename(columns={'index': 'year'})
    data = pd.melt(data, id_vars=['year'], var_name='Region', value_name='Wartość')
    data['year'] = pd.to_numeric(data['year'], errors='coerce')
    data = data.dropna(subset=['year']).astype({'year': int})

#     chart = alt.Chart(data).mark_line().encode(
#         x=alt.X('year:O',title='Rok'),y=alt.Y('Wartość:Q', title='Wartość'),color=alt.Color('Region:N',title='Wielkość'))
    chart = alt.Chart(data).mark_line().encode(
        x=alt.X('year:O', title='Rok'),
        y=alt.Y('Wartość:Q', title='Wartość'),
        color=alt.Color('Region:N', title='Wielkość')
    ).properties(
        title=alt.TitleParams(
            text='Wartość nakładów na sprzęt w latach 2015-2022',
            anchor='middle'))
    st.altair_chart(chart, use_container_width=True)

    st.write("### Wskaźnik nachylenia Sena")
    slope_data = []
    for region in countries:
        region_data = data[data['Region'] == region]
        slope, intercept, _, _, _ = linregress(region_data['year'], region_data['Wartość'])
        slope_data.append({'Region': region, 'Slope': slope, 'Intercept': intercept})
    sens_slope_results = pd.DataFrame(slope_data)
    st.write(sens_slope_results)

    st.write("### Test Manna-Kendalla")
    mk_data = []
    for region in countries:
        region_data = data[data['Region'] == region]
        mk_result = mk.original_test(region_data['Wartość'].values)
        mk_data.append({'Region': region, 'Trend': mk_result.trend, 'p-value': mk_result.p})
    mann_kendall_results = pd.DataFrame(mk_data)
    st.write(mann_kendall_results)
    
def data_visualization_wart_nakladow3():
    st.markdown('### Wartość nakładów na sprzęt informatyczny')

    @st.cache_data
    def get_local_data():
        try:
            df = pd.read_csv('transformed_data_wart_nakladow_sprzet_inf_wielkosc.csv')
            return df.set_index('Region')
        except Exception as e:
            st.error(f'Error loading data: {e}')
            return pd.DataFrame()

    df = get_local_data()
    if df.empty:
        return

    countries = st.multiselect('Wybierz wielkość przedsiębiorstw', list(df.index))
    if not countries:
        st.error('Wybierz co najmniej jedną.')
        return

    data = df.loc[countries].T.reset_index().rename(columns={'index': 'year'})
    data = pd.melt(data, id_vars=['year'], var_name='Region', value_name='Wartość')
    data['year'] = pd.to_numeric(data['year'], errors='coerce')
    data = data.dropna(subset=['year']).astype({'year': int})

#     chart = alt.Chart(data).mark_line().encode(
#         x=alt.X('year:O',title='Rok'),y=alt.Y('Wartość:Q', title='Wartość'),color=alt.Color('Region:N',title='Wielkość'))
    chart = alt.Chart(data).mark_line().encode(
        x=alt.X('year:O', title='Rok'),
        y=alt.Y('Wartość:Q', title='Wartość'),
        color=alt.Color('Region:N', title='Wielkość')
    ).properties(
        title=alt.TitleParams(
            text='Wartość nakładów na sprzęt informatyczny w latach 2015-2022',
            anchor='middle'))
    st.altair_chart(chart, use_container_width=True)

    st.write("### Wskaźnik nachylenia Sena")
    slope_data = []
    for region in countries:
        region_data = data[data['Region'] == region]
        slope, intercept, _, _, _ = linregress(region_data['year'], region_data['Wartość'])
        slope_data.append({'Region': region, 'Slope': slope, 'Intercept': intercept})
    sens_slope_results = pd.DataFrame(slope_data)
    st.write(sens_slope_results)

    st.write("### Test Manna-Kendalla")
    mk_data = []
    for region in countries:
        region_data = data[data['Region'] == region]
        mk_result = mk.original_test(region_data['Wartość'].values)
        mk_data.append({'Region': region, 'Trend': mk_result.trend, 'p-value': mk_result.p})
    mann_kendall_results = pd.DataFrame(mk_data)
    st.write(mann_kendall_results)

def data_visualization_wart_nakladow4():
    st.markdown('### Wartość nakładów na sprzęt telekomunikacyjny')

    @st.cache_data
    def get_local_data():
        try:
            df = pd.read_csv('transformed_data_wart_nakladow_sprzet_teleinf_wielkosc.csv')
            return df.set_index('Region')
        except Exception as e:
            st.error(f'Error loading data: {e}')
            return pd.DataFrame()

    df = get_local_data()
    if df.empty:
        return

    countries = st.multiselect('Wybierz wielkość przedsiębiorstw', list(df.index))
    if not countries:
        st.error('Wybierz co najmniej jedną.')
        return

    data = df.loc[countries].T.reset_index().rename(columns={'index': 'year'})
    data = pd.melt(data, id_vars=['year'], var_name='Region', value_name='Wartość')
    data['year'] = pd.to_numeric(data['year'], errors='coerce')
    data = data.dropna(subset=['year']).astype({'year': int})

#     chart = alt.Chart(data).mark_line().encode(
#         x=alt.X('year:O',title='Rok'),y=alt.Y('Wartość:Q', title='Wartość'),color=alt.Color('Region:N',title='Wielkość'))
    chart = alt.Chart(data).mark_line().encode(
        x=alt.X('year:O', title='Rok'),
        y=alt.Y('Wartość:Q', title='Wartość'),
        color=alt.Color('Region:N', title='Wielkość')
    ).properties(
        title=alt.TitleParams(
            text='Wartość nakładów na sprzęt telekomunikacyjny w latach 2015-2022',
            anchor='middle'))
    st.altair_chart(chart, use_container_width=True)

    st.write("### Wskaźnik nachylenia Sena")
    slope_data = []
    for region in countries:
        region_data = data[data['Region'] == region]
        slope, intercept, _, _, _ = linregress(region_data['year'], region_data['Wartość'])
        slope_data.append({'Region': region, 'Slope': slope, 'Intercept': intercept})
    sens_slope_results = pd.DataFrame(slope_data)
    st.write(sens_slope_results)

    st.write("### Test Manna-Kendalla")
    mk_data = []
    for region in countries:
        region_data = data[data['Region'] == region]
        mk_result = mk.original_test(region_data['Wartość'].values)
        mk_data.append({'Region': region, 'Trend': mk_result.trend, 'p-value': mk_result.p})
    mann_kendall_results = pd.DataFrame(mk_data)
    st.write(mann_kendall_results)

def data_visualization_wart_nakladow5():
    st.markdown('### Wartość nakładów na leasing urządzeń ICT')

    @st.cache_data
    def get_local_data():
        try:
            df = pd.read_csv('transformed_data_wart_nakladow_leasing_wielkosc.csv')
            return df.set_index('Region')
        except Exception as e:
            st.error(f'Error loading data: {e}')
            return pd.DataFrame()

    df = get_local_data()
    if df.empty:
        return

    countries = st.multiselect('Wybierz wielkość przedsiębiorstw', list(df.index))
    if not countries:
        st.error('Wybierz co najmniej jedną.')
        return

    data = df.loc[countries].T.reset_index().rename(columns={'index': 'year'})
    data = pd.melt(data, id_vars=['year'], var_name='Region', value_name='Wartość')
    data['year'] = pd.to_numeric(data['year'], errors='coerce')
    data = data.dropna(subset=['year']).astype({'year': int})

#     chart = alt.Chart(data).mark_line().encode(
#         x=alt.X('year:O',title='Rok'),y=alt.Y('Wartość:Q', title='Wartość'),color=alt.Color('Region:N',title='Wielkość'))
    chart = alt.Chart(data).mark_line().encode(
        x=alt.X('year:O', title='Rok'),
        y=alt.Y('Wartość:Q', title='Wartość'),
        color=alt.Color('Region:N', title='Wielkość')
    ).properties(
        title=alt.TitleParams(
            text='Wartość nakładów na leasing urządzeń ICT w latach 2015-2022',
            anchor='middle'))
    
    st.altair_chart(chart, use_container_width=True)

    st.write("### Wskaźnik nachylenia Sena")
    slope_data = []
    for region in countries:
        region_data = data[data['Region'] == region]
        slope, intercept, _, _, _ = linregress(region_data['year'], region_data['Wartość'])
        slope_data.append({'Region': region, 'Slope': slope, 'Intercept': intercept})
    sens_slope_results = pd.DataFrame(slope_data)
    st.write(sens_slope_results)

    st.write("### Test Manna-Kendalla")
    mk_data = []
    for region in countries:
        region_data = data[data['Region'] == region]
        mk_result = mk.original_test(region_data['Wartość'].values)
        mk_data.append({'Region': region, 'Trend': mk_result.trend, 'p-value': mk_result.p})
    mann_kendall_results = pd.DataFrame(mk_data)
    st.write(mann_kendall_results)

def data_visualization_wart_nakladow6():
    st.markdown('### Wartość nakładów na oprogramowanie')

    @st.cache_data
    def get_local_data():
        try:
            df = pd.read_csv('transformed_data_wart_nakladow_oprog_wielkosc.csv')
            return df.set_index('Region')
        except Exception as e:
            st.error(f'Error loading data: {e}')
            return pd.DataFrame()

    df = get_local_data()
    if df.empty:
        return

    countries = st.multiselect('Wybierz wielkość przedsiębiorstw', list(df.index))
    if not countries:
        st.error('Wybierz co najmniej jedną.')
        return

    data = df.loc[countries].T.reset_index().rename(columns={'index': 'year'})
    data = pd.melt(data, id_vars=['year'], var_name='Region', value_name='Wartość')
    data['year'] = pd.to_numeric(data['year'], errors='coerce')
    data = data.dropna(subset=['year']).astype({'year': int})

#     chart = alt.Chart(data).mark_line().encode(
#         x=alt.X('year:O',title='Rok'),y=alt.Y('Wartość:Q', title='Wartość'),color=alt.Color('Region:N',title='Wielkość'))
    chart = alt.Chart(data).mark_line().encode(
        x=alt.X('year:O', title='Rok'),
        y=alt.Y('Wartość:Q', title='Wartość'),
        color=alt.Color('Region:N', title='Wielkość')
    ).properties(
        title=alt.TitleParams(
            text='Wartość nakładów na oprogramowanie w latach 2015-2022',
            anchor='middle'))
    st.altair_chart(chart, use_container_width=True)

    st.write("### Wskaźnik nachylenia Sena")
    slope_data = []
    for region in countries:
        region_data = data[data['Region'] == region]
        slope, intercept, _, _, _ = linregress(region_data['year'], region_data['Wartość'])
        slope_data.append({'Region': region, 'Slope': slope, 'Intercept': intercept})
    sens_slope_results = pd.DataFrame(slope_data)
    st.write(sens_slope_results)

    st.write("### Test Manna-Kendalla")
    mk_data = []
    for region in countries:
        region_data = data[data['Region'] == region]
        mk_result = mk.original_test(region_data['Wartość'].values)
        mk_data.append({'Region': region, 'Trend': mk_result.trend, 'p-value': mk_result.p})
    mann_kendall_results = pd.DataFrame(mk_data)
    st.write(mann_kendall_results)
    
def rodzaje_polaczen_section2():
    st.sidebar.header("Rodzaje Połączeń")
    pages = {
        "szerokopasmowy  dostęp do Internetu": data_visualization_rodz_pol2,
        "szerokopasmowy  dostęp do Internetu poprzez łącze DSL": data_visualization_rodz_pol3,
    }
    selected_page = st.sidebar.selectbox("Wybierz analizę", pages.keys())
    pages[selected_page]()

def data_visualization_rodz_pol():

    @st.cache_data
    def get_local_data():
        try:
            df = pd.read_csv('transformed_data_rodz_pol_przed_o_wielkosc.csv')
            return df.set_index('Region')
        except Exception as e:
            st.error(f'Error loading data: {e}')
            return pd.DataFrame()

    df = get_local_data()
    if df.empty:
        return

    countries = st.multiselect('Wybierz wielkość przedsiębiorstw', list(df.index))
    if not countries:
        st.error('Wybierz co najmniej jedną.')
        return

    data = df.loc[countries].T.reset_index().rename(columns={'index': 'year'})
    data = pd.melt(data, id_vars=['year'], var_name='Region', value_name='Wartość')
    data['year'] = pd.to_numeric(data['year'], errors='coerce')
    data = data.dropna(subset=['year']).astype({'year': int})

#     chart = alt.Chart(data).mark_line().encode(
#         x=alt.X('year:O',title='Rok'),y=alt.Y('Wartość:Q', title='Wartość'),color=alt.Color('Region:N',title='Wielkość'))
    chart = alt.Chart(data).mark_line().encode(
        x=alt.X('year:O', title='Rok'),
        y=alt.Y('Wartość:Q', title='Wartość'),
        color=alt.Color('Region:N', title='Wielkość')
    ).properties(
        title=alt.TitleParams(
            text='Wartość nakładów na oprogramowanie w latach 2015-2022',
            anchor='middle'))
    st.altair_chart(chart, use_container_width=True)
    
    st.write("### Wskaźnik nachylenia Sena")
    slope_data = []
    for region in countries:
        region_data = data[data['Region'] == region]
        slope, intercept, _, _, _ = linregress(region_data['year'], region_data['Wartość'])
        slope_data.append({'Region': region, 'Slope': slope, 'Intercept': intercept})
    sens_slope_results = pd.DataFrame(slope_data)
    st.write(sens_slope_results)

    st.write("### Test Manna-Kendalla")
    mk_data = []
    for region in countries:
        region_data = data[data['Region'] == region]
        mk_result = mk.original_test(region_data['Wartość'].values)
        mk_data.append({'Region': region, 'Trend': mk_result.trend, 'p-value': mk_result.p})
    mann_kendall_results = pd.DataFrame(mk_data)
    st.write(mann_kendall_results)
    
def data_visualization_rodz_pol2():
    st.markdown('### Odsetek przedsiębiorstw z szerokopasmowym łączem internetowym względem przedsiębiorstw ogółem')

    @st.cache_data
    def get_data():
        try:
            df_total = pd.read_csv('transformed_data_rodz_pol_przed_o_wielkosc.csv').set_index('Region')
            df_partial = pd.read_csv('transformed_data_rodz_pol_szer_o_wielkosc.csv').set_index('Region')
            return df_total, df_partial
        except Exception as e:
            st.error(f'Error loading data: {e}')
            return pd.DataFrame(), pd.DataFrame()

    df_total, df_partial = get_data()
    if df_total.empty or df_partial.empty:
        return

    countries = st.multiselect('Wybierz wielkość przedsiębiorstw', list(df_partial.index))
    if not countries:
        st.error('Wybierz co najmniej jedną.')
        return

    df_percentage = (df_partial / df_total.replace(0, np.nan)) * 100
    data = df_percentage.loc[countries].T.reset_index().rename(columns={'index': 'year'})
    data = pd.melt(data, id_vars=['year'], var_name='Region', value_name='Procent')
    data['year'] = pd.to_numeric(data['year'], errors='coerce')
    data = data.dropna(subset=['year']).astype({'year': int})

#     chart = alt.Chart(data).mark_line().encode(
#         x=alt.X('year:O',title='Rok'),y=alt.Y('Procent:Q', title='Procent (%)'),color=alt.Color('Region:N',title='Wielkość'))
    chart = alt.Chart(data).mark_line().encode(
        x=alt.X('year:O', title='Rok'),
        y=alt.Y('Procent:Q', title='Procent (%)'),
        color=alt.Color('Region:N', title='Wielkość')
    ).properties(
        title=alt.TitleParams(
            text='Odsetek przedsiębiorstw z szerokopasmowym łączem internetowym w latach 2015-2022',
            anchor='middle', fontSize=12))
    st.altair_chart(chart, use_container_width=True)

    st.write("### Wskaźnik nachylenia Sena")
    slope_data = []
    for region in countries:
        region_data = data[data['Region'] == region].dropna(subset=['Procent'])
        if len(region_data) > 1:  
            slope, intercept, _, _, _ = linregress(region_data['year'], region_data['Procent'])
            slope_data.append({'Region': region, 'Slope': slope, 'Intercept': intercept})
    sens_slope_results = pd.DataFrame(slope_data)
    st.write(sens_slope_results)

    st.write("### Test Manna-Kendalla")
    mk_data = []
    for region in countries:
        region_data = data[data['Region'] == region].dropna(subset=['Procent'])
        if len(region_data) > 1:
            mk_result = mk.original_test(region_data['Procent'].values)
            mk_data.append({'Region': region, 'Trend': mk_result.trend, 'p-value': mk_result.p})
    mann_kendall_results = pd.DataFrame(mk_data)
    st.write(mann_kendall_results)
    
def data_visualization_rodz_pol3():
    st.markdown('### Odsetek połączenia DSL wśród łącza szerokopasmowego ogółem')

    @st.cache_data
    def get_data():
        try:
            df_total = pd.read_csv('transformed_data_rodz_pol_szer_o_wielkosc.csv').set_index('Region')
            df_partial = pd.read_csv('transformed_data_rodz_pol_szerDSL_wielkosc.csv').set_index('Region')
            return df_total, df_partial
        except Exception as e:
            st.error(f'Error loading data: {e}')
            return pd.DataFrame(), pd.DataFrame()

    df_total, df_partial = get_data()
    if df_total.empty or df_partial.empty:
        return

    countries = st.multiselect('Wybierz wielkość przedsiębiorstw', list(df_partial.index))
    if not countries:
        st.error('Wybierz co najmniej jedną.')
        return

    df_percentage = (df_partial / df_total.replace(0, np.nan)) * 100
    data = df_percentage.loc[countries].T.reset_index().rename(columns={'index': 'year'})
    data = pd.melt(data, id_vars=['year'], var_name='Region', value_name='Procent')
    data['year'] = pd.to_numeric(data['year'], errors='coerce')
    data = data.dropna(subset=['year']).astype({'year': int})

#     chart = alt.Chart(data).mark_line().encode(
#         x=alt.X('year:O',title='Rok'),y=alt.Y('Procent:Q', title='Procent (%)'),color=alt.Color('Region:N',title='Wielkość'))
    chart = alt.Chart(data).mark_line().encode(
        x=alt.X('year:O', title='Rok'),
        y=alt.Y('Procent:Q', title='Procent (%)'),
        color=alt.Color('Region:N', title='Wielkość')
    ).properties(
        title=alt.TitleParams(
            text='Odsetek połączeń DSL wśród łącz szerokopasmowych w latach 2015-2022',
            anchor='middle'))
    st.altair_chart(chart, use_container_width=True)

    st.write("### Wskaźnik nachylenia Sena")
    slope_data = []
    for region in countries:
        region_data = data[data['Region'] == region].dropna(subset=['Procent'])
        if len(region_data) > 1: 
            slope, intercept, _, _, _ = linregress(region_data['year'], region_data['Procent'])
            slope_data.append({'Region': region, 'Slope': slope, 'Intercept': intercept})
    sens_slope_results = pd.DataFrame(slope_data)
    st.write(sens_slope_results)

    st.write("### Test Manna-Kendalla")
    mk_data = []
    for region in countries:
        region_data = data[data['Region'] == region].dropna(subset=['Procent'])
        if len(region_data) > 1:
            mk_result = mk.original_test(region_data['Procent'].values)
            mk_data.append({'Region': region, 'Trend': mk_result.trend, 'p-value': mk_result.p})
    mann_kendall_results = pd.DataFrame(mk_data)
    st.write(mann_kendall_results)

def predkosc_polaczen_section2():
    st.sidebar.header("Prędkość Połączeń")
    pages = {
        "Przedsiębiorstwa z dostępem do internetu": visualization_pred_pol2,
        "szerokopasmowy  dostęp do Internetu - prędkość połączenia mniej niż 100 Mbit/s": visualization_pred_pol4,
        "szerokopasmowy  dostęp do Internetu - prędkość połączenia przynajmniej 100 Mbit/s":visualization_pred_pol5,
    }
    selected_page = st.sidebar.selectbox("Wybierz analizę", pages.keys())
    pages[selected_page]()
    
def visualization_pred_pol2():
    st.markdown('### Odsetek przedsiębiorstw z dostępem do internetu względem przedsiębiorstw ogółem')

    @st.cache_data
    def get_local_data():
        try:
            df_total = pd.read_excel('pred_pol_przed_wielkosc.xlsx').set_index('Region')
            df_partial = pd.read_excel('pred_pol_przed_di_wielkosc.xlsx').set_index('Region')
            return df_total, df_partial
        except Exception as e:
            st.error(f'Error loading data: {e}')
            return pd.DataFrame(), pd.DataFrame()

    df_total, df_partial = get_local_data()
    if df_total.empty:
        return

    countries = st.multiselect('Wybierz wielkość przedsiębiorstw', list(df_partial.index))
    if not countries:
        st.error('Wybierz co najmniej jedną.')
        return

    df_percentage = (df_partial / df_total.replace(0, np.nan)) * 100
    data = df_percentage.loc[countries].T.reset_index().rename(columns={'index': 'year'})
    data = pd.melt(data, id_vars=['year'], var_name='Region', value_name='Procent')
    data['year'] = pd.to_numeric(data['year'], errors='coerce')
    data = data.dropna(subset=['year']).astype({'year': int})

#     chart = alt.Chart(data).mark_line().encode(
#         x=alt.X('year:O',title='Rok'),y=alt.Y('Procent:Q', title='Procent (%)'),color=alt.Color('Region:N',title='Wielkość'))
    chart = alt.Chart(data).mark_line().encode(
        x=alt.X('year:O', title='Rok'),
        y=alt.Y('Procent:Q', title='Procent (%)'),
        color=alt.Color('Region:N', title='Wielkość')
    ).properties(
        title=alt.TitleParams(
            text='Odsetek przedsiębiorstw z dostępem do internetu w latach 2015-2022',
            anchor='middle'))
    st.altair_chart(chart, use_container_width=True)

    st.write("### Wskaźnik nachylenia Sena")
    slope_data = []
    for region in countries:
        region_data = data[data['Region'] == region].dropna(subset=['Procent'])
        if len(region_data) > 1:  
            slope, intercept, _, _, _ = linregress(region_data['year'], region_data['Procent'])
            slope_data.append({'Region': region, 'Slope': slope, 'Intercept': intercept})
    sens_slope_results = pd.DataFrame(slope_data)
    st.write(sens_slope_results)

    st.write("### Test Manna-Kendalla")
    mk_data = []
    for region in countries:
        region_data = data[data['Region'] == region].dropna(subset=['Procent'])
        if len(region_data) > 1:
            mk_result = mk.original_test(region_data['Procent'].values)
            mk_data.append({'Region': region, 'Trend': mk_result.trend, 'p-value': mk_result.p})
    mann_kendall_results = pd.DataFrame(mk_data)
    st.write(mann_kendall_results)

def visualization_pred_pol4():
    st.markdown('### Odsetek przedsiębiorstw z szerokopasmowym dostępem do Internetu o prędkości połączenia mniej niż 100 Mbit/s')

    @st.cache_data
    def get_local_data():
        try:
            df_total = pd.read_excel('pred_pol_szer_wielkosc.xlsx').set_index('Region')
            df_partial = pd.read_excel('pred_pol_szer_w_wielkosc.xlsx').set_index('Region')
            return df_total, df_partial
        except Exception as e:
            st.error(f'Error loading data: {e}')
            return pd.DataFrame(), pd.DataFrame()

    df_total, df_partial = get_local_data()
    if df_total.empty:
        return

    countries = st.multiselect('Wybierz wielkość przedsiębiorstw', list(df_partial.index))
    if not countries:
        st.error('Wybierz co najmniej jedną.')
        return

    df_percentage = (df_partial / df_total.replace(0, np.nan)) * 100
    data = df_percentage.loc[countries].T.reset_index().rename(columns={'index': 'year'})
    data = pd.melt(data, id_vars=['year'], var_name='Region', value_name='Procent')
    data['year'] = pd.to_numeric(data['year'], errors='coerce')
    data = data.dropna(subset=['year']).astype({'year': int})

#     chart = alt.Chart(data).mark_line().encode(
#         x=alt.X('year:O',title='Rok'),y=alt.Y('Procent:Q', title='Procent (%)'),color=alt.Color('Region:N',title='Wielkość'))
    chart = alt.Chart(data).mark_line().encode(
        x=alt.X('year:O', title='Rok'),
        y=alt.Y('Procent:Q', title='Procent (%)'),
        color=alt.Color('Region:N', title='Wielkość')
    ).properties(
        title=alt.TitleParams(
            text='Odsetek przedsiębiorstw z szerokopasmowym dostępem do Internetu o prędkości połączenia mniej niż 100 Mbit/s w latach 2015-2022',
            anchor='middle', fontSize=12))
    st.altair_chart(chart, use_container_width=True)

    st.write("### Wskaźnik nachylenia Sena")
    slope_data = []
    for region in countries:
        region_data = data[data['Region'] == region].dropna(subset=['Procent'])
        if len(region_data) > 1:  
            slope, intercept, _, _, _ = linregress(region_data['year'], region_data['Procent'])
            slope_data.append({'Region': region, 'Slope': slope, 'Intercept': intercept})
    sens_slope_results = pd.DataFrame(slope_data)
    st.write(sens_slope_results)

    st.write("### Test Manna-Kendalla")
    mk_data = []
    for region in countries:
        region_data = data[data['Region'] == region].dropna(subset=['Procent'])
        if len(region_data) > 1:
            mk_result = mk.original_test(region_data['Procent'].values)
            mk_data.append({'Region': region, 'Trend': mk_result.trend, 'p-value': mk_result.p})
    mann_kendall_results = pd.DataFrame(mk_data)
    st.write(mann_kendall_results)
    
def visualization_pred_pol5():
    st.markdown('### Odsetek przedsiębiorstw z szerokopasmowym dostęp do Internetu - prędkość połączenia przynajmniej 100 Mbit/s')

    @st.cache_data
    def get_local_data():
        try:
            df_total = pd.read_excel('pred_pol_szer_wielkosc.xlsx').set_index('Region')
            df_partial = pd.read_excel('pred_pol_szer_s_wielkosc.xlsx').set_index('Region')
            return df_total, df_partial
        except Exception as e:
            st.error(f'Error loading data: {e}')
            return pd.DataFrame(), pd.DataFrame()

    df_total, df_partial = get_local_data()
    if df_total.empty:
        return

    countries = st.multiselect('Wybierz wielkość przedsiębiorstw', list(df_partial.index))
    if not countries:
        st.error('Wybierz co najmniej jedną.')
        return

    df_percentage = (df_partial / df_total.replace(0, np.nan)) * 100
    data = df_percentage.loc[countries].T.reset_index().rename(columns={'index': 'year'})
    data = pd.melt(data, id_vars=['year'], var_name='Region', value_name='Procent')
    data['year'] = pd.to_numeric(data['year'], errors='coerce')
    data = data.dropna(subset=['year']).astype({'year': int})

#     chart = alt.Chart(data).mark_line().encode(
#         x=alt.X('year:O',title='Rok'),y=alt.Y('Procent:Q', title='Procent (%)'),color=alt.Color('Region:N',title='Wielkość'))
    chart = alt.Chart(data).mark_line().encode(
        x=alt.X('year:O', title='Rok'),
        y=alt.Y('Procent:Q', title='Procent (%)'),
        color=alt.Color('Region:N', title='Wielkość')
    ).properties(
        title=alt.TitleParams(
            text='Odsetek przedsiębiorstw z szerokopasmowym dostęp do Internetu - prędkość połączenia przynajmniej 100 Mbit/s w latach 2015-2022',
            anchor='middle', fontSize=12))
    st.altair_chart(chart, use_container_width=True)

    st.write("### Wskaźnik nachylenia Sena")
    slope_data = []
    for region in countries:
        region_data = data[data['Region'] == region].dropna(subset=['Procent'])
        if len(region_data) > 1:  
            slope, intercept, _, _, _ = linregress(region_data['year'], region_data['Procent'])
            slope_data.append({'Region': region, 'Slope': slope, 'Intercept': intercept})
    sens_slope_results = pd.DataFrame(slope_data)
    st.write(sens_slope_results)

    st.write("### Test Manna-Kendalla")
    mk_data = []
    for region in countries:
        region_data = data[data['Region'] == region].dropna(subset=['Procent'])
        if len(region_data) > 1:
            mk_result = mk.original_test(region_data['Procent'].values)
            mk_data.append({'Region': region, 'Trend': mk_result.trend, 'p-value': mk_result.p})
    mann_kendall_results = pd.DataFrame(mk_data)
    st.write(mann_kendall_results)

################# BRANŻE ##################################################################################################################
def data_dzialy():
    selected_subsection = st.sidebar.selectbox("Wybierz kategorię", ["Nakłady", "Pracownicy", "Wartość Nakładów", "Rodzaje Połączeń","Prędkość Połączeń"])
    if selected_subsection == "Pracownicy":
        pracownicy_section3()
    elif selected_subsection == "Nakłady":
        naklady_section3()
    elif selected_subsection == "Wartość Nakładów":
        wartosc_nakladow_section3()
    elif selected_subsection == "Rodzaje Połączeń":
        rodzaje_polaczen_section3()
    elif selected_subsection == "Prędkość Połączeń":
        predkosc_polaczen_section3()
        
def predkosc_polaczen_section3():
    st.sidebar.header("Prędkość Połączeń")
    pages = {
        "szerokopasmowy  dostęp do Internetu - prędkość połączenia mniej niż 100 Mbit/s": visualization4_pred_pol4,
        "szerokopasmowy  dostęp do Internetu - prędkość połączenia przynajmniej 100 Mbit/s":visualization5_pred_pol5,
    }
    selected_page = st.sidebar.selectbox("Wybierz analizę", pages.keys())
    pages[selected_page]()

def visualization4_pred_pol4():
    st.markdown('#### Przedsiębiorstwa z dostępem do szerokopasmowego łącza internetowego - prędkość połączenia mniej niż 100 Mbit/s')

    @st.cache_data
    def get_local_data():
        try:
            df = pd.read_excel('pred_pol_szer_w_dzialy.xlsx')
            return df.set_index('Region')
        except Exception as e:
            st.error(f'Error loading data: {e}')
            return pd.DataFrame()

    df = get_local_data()
    if df.empty:
        return

    countries = st.multiselect('Wybierz regiony', list(df.index))
    if not countries:
        st.error('Wybierz co najmniej jeden region.')
        return

    data = df.loc[countries].T.reset_index().rename(columns={'index': 'year'})
    data = pd.melt(data, id_vars=['year'], var_name='Region', value_name='Wartość')
    data['year'] = pd.to_numeric(data['year'], errors='coerce')
    data = data.dropna(subset=['year']).astype({'year': int})

#     chart = alt.Chart(data).mark_line().encode(
#         x=alt.X('year:O',title='Rok'),y=alt.Y('Wartość:Q', title='Wartość'),color=alt.Color('Region:N',title='Branża'))
    chart = alt.Chart(data).mark_line().encode(
        x=alt.X('year:O', title='Rok'),
        y=alt.Y('Wartość:Q', title='Wartość'),
        color=alt.Color('Region:N', title='Branża')
    ).properties(
        title=alt.TitleParams(
            text='Przedsiębiorstwa z dostępem do szerokopasmowego łącza internetowego - prędkość połączenia mniej niż 100 Mbit/s w latach 2015-2022',
            anchor='middle',
        fontSize=10))
    st.altair_chart(chart, use_container_width=True)

    st.write("### Wskaźnik nachylenia Sena")
    slope_data = []
    for region in countries:
        region_data = data[data['Region'] == region]
        slope, intercept, _, _, _ = linregress(region_data['year'], region_data['Wartość'])
        slope_data.append({'Region': region, 'Slope': slope, 'Intercept': intercept})
    sens_slope_results = pd.DataFrame(slope_data)
    st.write(sens_slope_results)

    st.write("### Test Manna-Kendalla")
    mk_data = []
    for region in countries:
        region_data = data[data['Region'] == region]
        mk_result = mk.original_test(region_data['Wartość'].values)
        mk_data.append({'Region': region, 'Trend': mk_result.trend, 'p-value': mk_result.p})
    mann_kendall_results = pd.DataFrame(mk_data)
    st.write(mann_kendall_results)
    
def visualization5_pred_pol5():
    st.markdown('#### Przedsiębiorstwa z dostępem do szerokopasmowego łącza internetowego o prędkości <= 100 Mbit/s')

    @st.cache_data
    def get_local_data():
        try:
            df = pd.read_excel('pred_pol_szer_s_dzialy.xlsx')
            return df.set_index('Region')
        except Exception as e:
            st.error(f'Error loading data: {e}')
            return pd.DataFrame()

    df = get_local_data()
    if df.empty:
        return

    countries = st.multiselect('Wybierz regiony', list(df.index))
    if not countries:
        st.error('Wybierz co najmniej jeden region.')
        return

    data = df.loc[countries].T.reset_index().rename(columns={'index': 'year'})
    data = pd.melt(data, id_vars=['year'], var_name='Region', value_name='Wartość')
    data['year'] = pd.to_numeric(data['year'], errors='coerce')
    data = data.dropna(subset=['year']).astype({'year': int})

#     chart = alt.Chart(data).mark_line().encode(
#         x=alt.X('year:O',title='Rok'),y=alt.Y('Wartość:Q', title='Wartość'),color=alt.Color('Region:N',title='Branża'))
    chart = alt.Chart(data).mark_line().encode(
        x=alt.X('year:O', title='Rok'),
        y=alt.Y('Wartość:Q', title='Wartość'),
        color=alt.Color('Region:N', title='Branża')
    ).properties(
        title=alt.TitleParams(
            text='Przedsiębiorstwa z dostępem do szerokopasmowego łącza internetowego o prędkości >= 100 Mbit/s w latach 2015-2022',
            anchor='middle',fontSize=10))
    st.altair_chart(chart, use_container_width=True)

    st.write("### Wskaźnik nachylenia Sena")
    slope_data = []
    for region in countries:
        region_data = data[data['Region'] == region]
        slope, intercept, _, _, _ = linregress(region_data['year'], region_data['Wartość'])
        slope_data.append({'Region': region, 'Slope': slope, 'Intercept': intercept})
    sens_slope_results = pd.DataFrame(slope_data)
    st.write(sens_slope_results)

    st.write("### Test Manna-Kendalla")
    mk_data = []
    for region in countries:
        region_data = data[data['Region'] == region]
        mk_result = mk.original_test(region_data['Wartość'].values)
        mk_data.append({'Region': region, 'Trend': mk_result.trend, 'p-value': mk_result.p})
    mann_kendall_results = pd.DataFrame(mk_data)
    st.write(mann_kendall_results)

def pracownicy_section3():
    st.sidebar.header("Pracownicy")
    pages = {
        "Przedsiębiorstwa zatrudniające pracowników": data_dzialy4,
        "Pracujący ogółem": data_dzialy3,
        "Pracownicy z wyższym wykształceniem": data_dzialy1,
        "Pracownicy z dostępem do internetu": data_dzialy2,
        "Pracownicy z urządzeniami przenośnymi": data_dzialy5,
    }
    selected_page = st.sidebar.selectbox("Wybierz analizę", pages.keys())
    pages[selected_page]()    
    
def data_dzialy1():
    st.markdown('### Pracownicy z wyższym wykształceniem')

    @st.cache_data
    def get_local_data():
        try:
            df = pd.read_excel('pr_ww_dzialy.xlsx', sheet_name=0)  
            return df.set_index('Region')
        except Exception as e:
            st.error(f'Error loading data: {e}')
            return pd.DataFrame()

    df = get_local_data()
    if df.empty:
        return

    countries = st.multiselect('Wybierz dział', list(df.index))
    if not countries:
        st.error('Wybierz co najmniej jeden.')
        return

    data = df.loc[countries].T.reset_index().rename(columns={'index': 'year'})
    data = pd.melt(data, id_vars=['year'], var_name='Region', value_name='Wartość')
    data['year'] = pd.to_numeric(data['year'], errors='coerce')
    data = data.dropna(subset=['year']).astype({'year': int})

#     sens_chart = alt.Chart(data).mark_line().encode(
#        x=alt.X('year:O',title='Rok'),y=alt.Y('Wartość:Q', title='Wartość'),color=alt.Color('Region:N',title='Branża'))
    chart = alt.Chart(data).mark_line().encode(
        x=alt.X('year:O', title='Rok'),
        y=alt.Y('Wartość:Q', title='Wartość'),
        color=alt.Color('Region:N', title='Branża')
    ).properties(
        title=alt.TitleParams(
            text='Pracownicy z wyższym wykształceniem w latach 2015-2022',
            anchor='middle'))
    st.altair_chart(chart, use_container_width=True)

    st.write("### Wskaźnik nachylenia Sena")
    slope_data = []
    for region in countries:
        region_data = data[data['Region'] == region]
        slope, intercept, _, _, _ = linregress(region_data['year'], region_data['Wartość'])
        slope_data.append({'Region': region, 'Slope': slope, 'Intercept': intercept})
    sens_slope_results = pd.DataFrame(slope_data)
    st.write(sens_slope_results)

    st.write("### Test Manna-Kendalla")
    mk_data = []
    for region in countries:
        region_data = data[data['Region'] == region]
        mk_result = mk.original_test(region_data['Wartość'].values)
        mk_data.append({'Region': region, 'Trend': mk_result.trend, 'p-value': mk_result.p})
    mann_kendall_results = pd.DataFrame(mk_data)
    st.write(mann_kendall_results)

def data_dzialy2():
    st.markdown('### Pracownicy z dostępem do internetu')

    @st.cache_data
    def get_local_data():
        try:
            df = pd.read_excel('pr_di_dzialy.xlsx', sheet_name=0)
            return df.set_index('Region')
        except Exception as e:
            st.error(f'Error loading data: {e}')
            return pd.DataFrame()

    df = get_local_data()
    if df.empty:
        return

    countries = st.multiselect('Wybierz dział', list(df.index))
    if not countries:
        st.error('Wybierz co najmniej jeden.')
        return

    data = df.loc[countries].T.reset_index().rename(columns={'index': 'year'})
    data = pd.melt(data, id_vars=['year'], var_name='Region', value_name='Wartość')
    data['year'] = pd.to_numeric(data['year'], errors='coerce')
    data = data.dropna(subset=['year']).astype({'year': int})

#     sens_chart = alt.Chart(data).mark_line().encode(
#        x=alt.X('year:O',title='Rok'),y=alt.Y('Wartość:Q', title='Wartość'),color=alt.Color('Region:N',title='Branża'))
    chart = alt.Chart(data).mark_line().encode(
        x=alt.X('year:O', title='Rok'),
        y=alt.Y('Wartość:Q', title='Wartość'),
        color=alt.Color('Region:N', title='Branża')
    ).properties(
        title=alt.TitleParams(
            text='Pracownicy z dostępem do internetu w latach 2015-2022',
            anchor='middle'))
    st.altair_chart(chart, use_container_width=True)

    st.write("### Wskaźnik nachylenia Sena")
    slope_data = []
    for region in countries:
        region_data = data[data['Region'] == region]
        slope, intercept, _, _, _ = linregress(region_data['year'], region_data['Wartość'])
        slope_data.append({'Region': region, 'Slope': slope, 'Intercept': intercept})
    sens_slope_results = pd.DataFrame(slope_data)
    st.write(sens_slope_results)

    st.write("### Test Manna-Kendalla")
    mk_data = []
    for region in countries:
        region_data = data[data['Region'] == region]
        mk_result = mk.original_test(region_data['Wartość'].values)
        mk_data.append({'Region': region, 'Trend': mk_result.trend, 'p-value': mk_result.p})
    mann_kendall_results = pd.DataFrame(mk_data)
    st.write(mann_kendall_results)
    
def data_dzialy3():
    st.markdown('### Pracujący ogółem')

    @st.cache_data
    def get_local_data():
        try:
            df = pd.read_excel('pr_o_dzialy.xlsx', sheet_name=0) 
            return df.set_index('Region')
        except Exception as e:
            st.error(f'Error loading data: {e}')
            return pd.DataFrame()

    df = get_local_data()
    if df.empty:
        return

    countries = st.multiselect('Wybierz dział', list(df.index))
    if not countries:
        st.error('Wybierz co najmniej jeden.')
        return

    data = df.loc[countries].T.reset_index().rename(columns={'index': 'year'})
    data = pd.melt(data, id_vars=['year'], var_name='Region', value_name='Wartość')
    data['year'] = pd.to_numeric(data['year'], errors='coerce')
    data = data.dropna(subset=['year']).astype({'year': int})

#     sens_chart = alt.Chart(data).mark_line().encode(
#        x=alt.X('year:O',title='Rok'),y=alt.Y('Wartość:Q', title='Wartość'),color=alt.Color('Region:N',title='Branża'))
    chart = alt.Chart(data).mark_line().encode(
        x=alt.X('year:O', title='Rok'),
        y=alt.Y('Wartość:Q', title='Wartość'),
        color=alt.Color('Region:N', title='Branża')
    ).properties(
        title=alt.TitleParams(
            text='Pracujący ogółem w latach 2015-2022',
            anchor='middle'))
    st.altair_chart(chart, use_container_width=True)

    st.write("### Wskaźnik nachylenia Sena")
    slope_data = []
    for region in countries:
        region_data = data[data['Region'] == region]
        slope, intercept, _, _, _ = linregress(region_data['year'], region_data['Wartość'])
        slope_data.append({'Region': region, 'Slope': slope, 'Intercept': intercept})
    sens_slope_results = pd.DataFrame(slope_data)
    st.write(sens_slope_results)

    st.write("### Test Manna-Kendalla")
    mk_data = []
    for region in countries:
        region_data = data[data['Region'] == region]
        mk_result = mk.original_test(region_data['Wartość'].values)
        mk_data.append({'Region': region, 'Trend': mk_result.trend, 'p-value': mk_result.p})
    mann_kendall_results = pd.DataFrame(mk_data)
    st.write(mann_kendall_results)
    
def data_dzialy4():
    st.markdown('### Przedsiębiorstwa zatrudniające pracowników')

    @st.cache_data
    def get_local_data():
        try:
            df = pd.read_excel('pr_przed_o_dzialy.xlsx', sheet_name=0) 
            return df.set_index('Region')
        except Exception as e:
            st.error(f'Error loading data: {e}')
            return pd.DataFrame()

    df = get_local_data()
    if df.empty:
        return

    countries = st.multiselect('Wybierz dział', list(df.index))
    if not countries:
        st.error('Wybierz co najmniej jeden.')
        return

    data = df.loc[countries].T.reset_index().rename(columns={'index': 'year'})
    data = pd.melt(data, id_vars=['year'], var_name='Region', value_name='Wartość')
    data['year'] = pd.to_numeric(data['year'], errors='coerce')
    data = data.dropna(subset=['year']).astype({'year': int})

#     sens_chart = alt.Chart(data).mark_line().encode(
#        x=alt.X('year:O',title='Rok'),y=alt.Y('Wartość:Q', title='Wartość'),color=alt.Color('Region:N',title='Branża'))
    chart = alt.Chart(data).mark_line().encode(
        x=alt.X('year:O', title='Rok'),
        y=alt.Y('Wartość:Q', title='Wartość'),
        color=alt.Color('Region:N', title='Branża')
    ).properties(
        title=alt.TitleParams(
            text='Przedsiębiorstwa zatrudniające pracowników w latach 2015-2022',
            anchor='middle'))
    st.altair_chart(chart, use_container_width=True)

    st.write("### Wskaźnik nachylenia Sena")    
    slope_data = []
    for region in countries:
        region_data = data[data['Region'] == region]
        slope, intercept, _, _, _ = linregress(region_data['year'], region_data['Wartość'])
        slope_data.append({'Region': region, 'Slope': slope, 'Intercept': intercept})
    sens_slope_results = pd.DataFrame(slope_data)
    st.write(sens_slope_results)
        
    st.write("### Test Manna-Kendalla")
    mk_data = []
    for region in countries:
        region_data = data[data['Region'] == region]
        mk_result = mk.original_test(region_data['Wartość'].values)
        mk_data.append({'Region': region, 'Trend': mk_result.trend, 'p-value': mk_result.p})
    mann_kendall_results = pd.DataFrame(mk_data)
    st.write(mann_kendall_results)
    
def data_dzialy5():
    st.markdown('### Pracownicy z urządzeniami przenośnymi')

    @st.cache_data
    def get_local_data():
        try:
            df = pd.read_excel('pr_up_dzialy.xlsx', sheet_name=0) 
            return df.set_index('Region')
        except Exception as e:
            st.error(f'Error loading data: {e}')
            return pd.DataFrame()

    df = get_local_data()
    if df.empty:
        return

    countries = st.multiselect('Wybierz dział', list(df.index))
    if not countries:
        st.error('Wybierz co najmniej jeden.')
        return

    data = df.loc[countries].T.reset_index().rename(columns={'index': 'year'})
    data = pd.melt(data, id_vars=['year'], var_name='Region', value_name='Wartość')
    data['year'] = pd.to_numeric(data['year'], errors='coerce')
    data = data.dropna(subset=['year']).astype({'year': int})

#     sens_chart = alt.Chart(data).mark_line().encode(
#        x=alt.X('year:O',title='Rok'),y=alt.Y('Wartość:Q', title='Wartość'),color=alt.Color('Region:N',title='Branża'))
    chart = alt.Chart(data).mark_line().encode(
        x=alt.X('year:O', title='Rok'),
        y=alt.Y('Wartość:Q', title='Wartość'),
        color=alt.Color('Region:N', title='Branża')
    ).properties(
        title=alt.TitleParams(
            text='Pracownicy z urządzeniami przenośnymi w latach 2015-2022',
            anchor='middle'))
    st.altair_chart(chart, use_container_width=True)

    st.write("### Wskaźnik nachylenia Sena")
    slope_data = []
    for region in countries:
        region_data = data[data['Region'] == region]
        slope, intercept, _, _, _ = linregress(region_data['year'], region_data['Wartość'])
        slope_data.append({'Region': region, 'Slope': slope, 'Intercept': intercept})
    sens_slope_results = pd.DataFrame(slope_data)
    st.write(sens_slope_results)

    st.write("### Test Manna-Kendalla")
    mk_data = []
    for region in countries:
        region_data = data[data['Region'] == region]
        mk_result = mk.original_test(region_data['Wartość'].values)
        mk_data.append({'Region': region, 'Trend': mk_result.trend, 'p-value': mk_result.p})
    mann_kendall_results = pd.DataFrame(mk_data)
    st.write(mann_kendall_results)   

def naklady_section3():
    st.sidebar.header("Nakłady")
    pages = {
        "Nakłady na ICT w przedsiębiorstwach": data_naklady1,
        "Nakłady na sprzęt/oprogramowanie": data_naklady2,
        "Nakłady - sprzęt ogółem": data_naklady3,
        "Nakłady - sprzęt informatyczny": data_naklady4,
        "Nakłady - sprzęt teleinformatyczny": data_naklady5,
        "Nakłady - leasing": data_naklady6,
        "Nakłady - oprogramowanie": data_naklady7,
    }
    selected_page = st.sidebar.selectbox("Wybierz analizę", pages.keys())
    pages[selected_page]()
    
def data_naklady1():
    st.markdown('### Nakłady na ICT w przedsiębiorstwach')

    @st.cache_data
    def get_local_data():
        try:
            df = pd.read_excel('nak_przed_o_dzialy.xlsx', sheet_name=0) 
            return df.set_index('Region')
        except Exception as e:
            st.error(f'Error loading data: {e}')
            return pd.DataFrame()

    df = get_local_data()
    if df.empty:
        return

    countries = st.multiselect('Wybierz dział', list(df.index))
    if not countries:
        st.error('Wybierz co najmniej jeden.')
        return

    data = df.loc[countries].T.reset_index().rename(columns={'index': 'year'})
    data = pd.melt(data, id_vars=['year'], var_name='Region', value_name='Wartość')
    data['year'] = pd.to_numeric(data['year'], errors='coerce')
    data = data.dropna(subset=['year']).astype({'year': int})

#     sens_chart = alt.Chart(data).mark_line().encode(
#        x=alt.X('year:O',title='Rok'),y=alt.Y('Wartość:Q', title='Wartość'),color=alt.Color('Region:N',title='Branża'))
    chart = alt.Chart(data).mark_line().encode(
        x=alt.X('year:O', title='Rok'),
        y=alt.Y('Wartość:Q', title='Wartość'),
        color=alt.Color('Region:N', title='Branża')
    ).properties(
        title=alt.TitleParams(
            text='Nakłady na ICT w przedsiębiorstwach w latach 2015-2022',
            anchor='middle'))
    st.altair_chart(chart, use_container_width=True)

    st.write("### Wskaźnik nachylenia Sena")
    slope_data = []
    for region in countries:
        region_data = data[data['Region'] == region]
        slope, intercept, _, _, _ = linregress(region_data['year'], region_data['Wartość'])
        slope_data.append({'Region': region, 'Slope': slope, 'Intercept': intercept})
    sens_slope_results = pd.DataFrame(slope_data)
    st.write(sens_slope_results)

    st.write("### Test Manna-Kendalla")
    mk_data = []
    for region in countries:
        region_data = data[data['Region'] == region]
        mk_result = mk.original_test(region_data['Wartość'].values)
        mk_data.append({'Region': region, 'Trend': mk_result.trend, 'p-value': mk_result.p})
    mann_kendall_results = pd.DataFrame(mk_data)
    st.write(mann_kendall_results)
    
def data_naklady2():
    st.markdown('### Nakłady na sprzęt i oprogramowanie')

    @st.cache_data
    def get_local_data():
        try:
            df = pd.read_excel('nak_sprzet_opr_dzialy.xlsx', sheet_name=0) 
            return df.set_index('Region')
        except Exception as e:
            st.error(f'Error loading data: {e}')
            return pd.DataFrame()

    df = get_local_data()
    if df.empty:
        return

    countries = st.multiselect('Wybierz dział', list(df.index))
    if not countries:
        st.error('Wybierz co najmniej jeden.')
        return

    data = df.loc[countries].T.reset_index().rename(columns={'index': 'year'})
    data = pd.melt(data, id_vars=['year'], var_name='Region', value_name='Wartość')
    data['year'] = pd.to_numeric(data['year'], errors='coerce')
    data = data.dropna(subset=['year']).astype({'year': int})

#     sens_chart = alt.Chart(data).mark_line().encode(
#        x=alt.X('year:O',title='Rok'),y=alt.Y('Wartość:Q', title='Wartość'),color=alt.Color('Region:N',title='Branża'))
    chart = alt.Chart(data).mark_line().encode(
        x=alt.X('year:O', title='Rok'),
        y=alt.Y('Wartość:Q', title='Wartość'),
        color=alt.Color('Region:N', title='Branża')
    ).properties(
        title=alt.TitleParams(
            text='Nakłady na sprzęt i oprogramowanie w latach 2015-2022',
            anchor='middle'))
    st.altair_chart(chart, use_container_width=True)

    st.write("### Wskaźnik nachylenia Sena")
    slope_data = []
    for region in countries:
        region_data = data[data['Region'] == region]
        slope, intercept, _, _, _ = linregress(region_data['year'], region_data['Wartość'])
        slope_data.append({'Region': region, 'Slope': slope, 'Intercept': intercept})
    sens_slope_results = pd.DataFrame(slope_data)
    st.write(sens_slope_results)

    st.write("### Test Manna-Kendalla")
    mk_data = []
    for region in countries:
        region_data = data[data['Region'] == region]
        mk_result = mk.original_test(region_data['Wartość'].values)
        mk_data.append({'Region': region, 'Trend': mk_result.trend, 'p-value': mk_result.p})
    mann_kendall_results = pd.DataFrame(mk_data)
    st.write(mann_kendall_results)
    
def data_naklady3():
    st.markdown('### Nakłady na sprzęt ogółem')

    @st.cache_data
    def get_local_data():
        try:
            df = pd.read_excel('nak_sprzet_dzialy.xlsx', sheet_name=0) 
            return df.set_index('Region')
        except Exception as e:
            st.error(f'Error loading data: {e}')
            return pd.DataFrame()

    df = get_local_data()
    if df.empty:
        return

    countries = st.multiselect('Wybierz dział', list(df.index))
    if not countries:
        st.error('Wybierz co najmniej jeden.')
        return

    data = df.loc[countries].T.reset_index().rename(columns={'index': 'year'})
    data = pd.melt(data, id_vars=['year'], var_name='Region', value_name='Wartość')
    data['year'] = pd.to_numeric(data['year'], errors='coerce')
    data = data.dropna(subset=['year']).astype({'year': int})

#     sens_chart = alt.Chart(data).mark_line().encode(
#        x=alt.X('year:O',title='Rok'),y=alt.Y('Wartość:Q', title='Wartość'),color=alt.Color('Region:N',title='Branża'))
    chart = alt.Chart(data).mark_line().encode(
        x=alt.X('year:O', title='Rok'),
        y=alt.Y('Wartość:Q', title='Wartość'),
        color=alt.Color('Region:N', title='Branża')
    ).properties(
        title=alt.TitleParams(
            text='Nakłady na sprzęt w latach 2015-2022',
            anchor='middle'))
    st.altair_chart(chart, use_container_width=True)

    st.write("### Wskaźnik nachylenia Sena")
    slope_data = []
    for region in countries:
        region_data = data[data['Region'] == region]
        slope, intercept, _, _, _ = linregress(region_data['year'], region_data['Wartość'])
        slope_data.append({'Region': region, 'Slope': slope, 'Intercept': intercept})
    sens_slope_results = pd.DataFrame(slope_data)
    st.write(sens_slope_results)

    st.write("### Test Manna-Kendalla")
    mk_data = []
    for region in countries:
        region_data = data[data['Region'] == region]
        mk_result = mk.original_test(region_data['Wartość'].values)
        mk_data.append({'Region': region, 'Trend': mk_result.trend, 'p-value': mk_result.p})
    mann_kendall_results = pd.DataFrame(mk_data)
    st.write(mann_kendall_results)
    
def data_naklady4():
    st.markdown('### Nakłady na sprzęt informatyczny')

    @st.cache_data
    def get_local_data():
        try:
            df = pd.read_excel('nak_sprzet_inf_dzialy.xlsx', sheet_name=0)  
            return df.set_index('Region')
        except Exception as e:
            st.error(f'Error loading data: {e}')
            return pd.DataFrame()

    df = get_local_data()
    if df.empty:
        return

    countries = st.multiselect('Wybierz dział', list(df.index))
    if not countries:
        st.error('Wybierz co najmniej jeden.')
        return

    data = df.loc[countries].T.reset_index().rename(columns={'index': 'year'})
    data = pd.melt(data, id_vars=['year'], var_name='Region', value_name='Wartość')
    data['year'] = pd.to_numeric(data['year'], errors='coerce')
    data = data.dropna(subset=['year']).astype({'year': int})

#     sens_chart = alt.Chart(data).mark_line().encode(
#        x=alt.X('year:O',title='Rok'),y=alt.Y('Wartość:Q', title='Wartość'),color=alt.Color('Region:N',title='Branża'))
#     st.altair_chart(sens_chart, use_container_width=True)
    chart = alt.Chart(data).mark_line().encode(
        x=alt.X('year:O', title='Rok'),
        y=alt.Y('Wartość:Q', title='Wartość'),
        color=alt.Color('Region:N', title='Branża')
    ).properties(
        title=alt.TitleParams(
            text='Nakłady na sprzęt informatyczny w latach 2015-2022',
            anchor='middle'))
    st.altair_chart(chart, use_container_width=True)
    
    st.write("### Wskaźnik nachylenia Sena")
    slope_data = []
    for region in countries:
        region_data = data[data['Region'] == region]
        slope, intercept, _, _, _ = linregress(region_data['year'], region_data['Wartość'])
        slope_data.append({'Region': region, 'Slope': slope, 'Intercept': intercept})
    sens_slope_results = pd.DataFrame(slope_data)
    st.write(sens_slope_results)

    st.write("### Test Manna-Kendalla")
    mk_data = []
    for region in countries:
        region_data = data[data['Region'] == region]
        mk_result = mk.original_test(region_data['Wartość'].values)
        mk_data.append({'Region': region, 'Trend': mk_result.trend, 'p-value': mk_result.p})
    mann_kendall_results = pd.DataFrame(mk_data)
    st.write(mann_kendall_results)
    
def data_naklady5():
    st.markdown('### Nakłady na sprzęt telekomunikacyjny')

    @st.cache_data
    def get_local_data():
        try:
            df = pd.read_excel('nak_sprzet_tele_dzialy.xlsx', sheet_name=0) 
            return df.set_index('Region')
        except Exception as e:
            st.error(f'Error loading data: {e}')
            return pd.DataFrame()

    df = get_local_data()
    if df.empty:
        return

    countries = st.multiselect('Wybierz dział', list(df.index))
    if not countries:
        st.error('Wybierz co najmniej jeden.')
        return

    data = df.loc[countries].T.reset_index().rename(columns={'index': 'year'})
    data = pd.melt(data, id_vars=['year'], var_name='Region', value_name='Wartość')
    data['year'] = pd.to_numeric(data['year'], errors='coerce')
    data = data.dropna(subset=['year']).astype({'year': int})

#     sens_chart = alt.Chart(data).mark_line().encode(
#        x=alt.X('year:O',title='Rok'),y=alt.Y('Wartość:Q', title='Wartość'),color=alt.Color('Region:N',title='Branża'))
#     st.altair_chart(sens_chart, use_container_width=True)
    chart = alt.Chart(data).mark_line().encode(
        x=alt.X('year:O', title='Rok'),
        y=alt.Y('Wartość:Q', title='Wartość'),
        color=alt.Color('Region:N', title='Branża')
    ).properties(
        title=alt.TitleParams(
            text='Nakłady na sprzęt telekomunikacyjny w latach 2015-2022',
            anchor='middle'))
    st.altair_chart(chart, use_container_width=True)
    
    st.write("### Wskaźnik nachylenia Sena")
    slope_data = []
    for region in countries:
        region_data = data[data['Region'] == region]
        slope, intercept, _, _, _ = linregress(region_data['year'], region_data['Wartość'])
        slope_data.append({'Region': region, 'Slope': slope, 'Intercept': intercept})
    sens_slope_results = pd.DataFrame(slope_data)
    st.write(sens_slope_results)
    
    st.write("### Test Manna-Kendalla")
    mk_data = []
    for region in countries:
        region_data = data[data['Region'] == region]
        mk_result = mk.original_test(region_data['Wartość'].values)
        mk_data.append({'Region': region, 'Trend': mk_result.trend, 'p-value': mk_result.p})
    mann_kendall_results = pd.DataFrame(mk_data)
    st.write(mann_kendall_results)
    
def data_naklady6():
    st.markdown('### Nakłady na leasing urządzeń ICT')

    @st.cache_data
    def get_local_data():
        try:
            df = pd.read_excel('nak_leasing_dzialy.xlsx', sheet_name=0)
            return df.set_index('Region')
        except Exception as e:
            st.error(f'Error loading data: {e}')
            return pd.DataFrame()

    df = get_local_data()
    if df.empty:
        return

    countries = st.multiselect('Wybierz dział', list(df.index))
    if not countries:
        st.error('Wybierz co najmniej jeden.')
        return

    data = df.loc[countries].T.reset_index().rename(columns={'index': 'year'})
    data = pd.melt(data, id_vars=['year'], var_name='Region', value_name='Wartość')
    data['year'] = pd.to_numeric(data['year'], errors='coerce')
    data = data.dropna(subset=['year']).astype({'year': int})

#     sens_chart = alt.Chart(data).mark_line().encode(
#        x=alt.X('year:O',title='Rok'),y=alt.Y('Wartość:Q', title='Wartość'),color=alt.Color('Region:N',title='Branża'))
#     st.altair_chart(sens_chart, use_container_width=True)
    chart = alt.Chart(data).mark_line().encode(
        x=alt.X('year:O', title='Rok'),
        y=alt.Y('Wartość:Q', title='Wartość'),
        color=alt.Color('Region:N', title='Branża')
    ).properties(
        title=alt.TitleParams(
            text='Nakłady na leasing urządzeń ICT w latach 2015-2022',
            anchor='middle'))
    st.altair_chart(chart, use_container_width=True)
    
    st.write("### Wskaźnik nachylenia Sena")
    slope_data = []
    for region in countries:
        region_data = data[data['Region'] == region]
        slope, intercept, _, _, _ = linregress(region_data['year'], region_data['Wartość'])
        slope_data.append({'Region': region, 'Slope': slope, 'Intercept': intercept})
    sens_slope_results = pd.DataFrame(slope_data)
    st.write(sens_slope_results)

    st.write("### Test Manna-Kendalla")
    mk_data = []
    for region in countries:
        region_data = data[data['Region'] == region]
        mk_result = mk.original_test(region_data['Wartość'].values)
        mk_data.append({'Region': region, 'Trend': mk_result.trend, 'p-value': mk_result.p})
    mann_kendall_results = pd.DataFrame(mk_data)
    st.write(mann_kendall_results)
    
def data_naklady7():
    st.markdown('### Nakłady na oprogramowanie')

    @st.cache_data
    def get_local_data():
        try:
            df = pd.read_excel('nak_oprog_dzialy.xlsx', sheet_name=0) 
            return df.set_index('Region')
        except Exception as e:
            st.error(f'Error loading data: {e}')
            return pd.DataFrame()

    df = get_local_data()
    if df.empty:
        return

    countries = st.multiselect('Wybierz dział', list(df.index))
    if not countries:
        st.error('Wybierz co najmniej jeden.')
        return

    data = df.loc[countries].T.reset_index().rename(columns={'index': 'year'})
    data = pd.melt(data, id_vars=['year'], var_name='Region', value_name='Wartość')
    data['year'] = pd.to_numeric(data['year'], errors='coerce')
    data = data.dropna(subset=['year']).astype({'year': int})

#     sens_chart = alt.Chart(data).mark_line().encode(
#        x=alt.X('year:O',title='Rok'),y=alt.Y('Wartość:Q', title='Wartość'),color=alt.Color('Region:N',title='Branża'))
#     st.altair_chart(sens_chart, use_container_width=True)
    chart = alt.Chart(data).mark_line().encode(
        x=alt.X('year:O', title='Rok'),
        y=alt.Y('Wartość:Q', title='Wartość'),
        color=alt.Color('Region:N', title='Branża')
    ).properties(
        title=alt.TitleParams(
            text='Nakłady na oprogramowanie w latach 2015-2022',
            anchor='middle'))
    st.altair_chart(chart, use_container_width=True)
    
    st.write("### Wskaźnik nachylenia Sena")
    slope_data = []
    for region in countries:
        region_data = data[data['Region'] == region]
        slope, intercept, _, _, _ = linregress(region_data['year'], region_data['Wartość'])
        slope_data.append({'Region': region, 'Slope': slope, 'Intercept': intercept})
    sens_slope_results = pd.DataFrame(slope_data)
    st.write(sens_slope_results)

    st.write("### Test Manna-Kendalla")
    mk_data = []
    for region in countries:
        region_data = data[data['Region'] == region]
        mk_result = mk.original_test(region_data['Wartość'].values)
        mk_data.append({'Region': region, 'Trend': mk_result.trend, 'p-value': mk_result.p})
    mann_kendall_results = pd.DataFrame(mk_data)
    st.write(mann_kendall_results)

def wartosc_nakladow_section3():
    st.sidebar.header("Wartość Nakładów")
    pages = {
        "Wartość nakładów - sprzęt/oprogramowanie": data_wart_nakladow1,
        "Wartość nakładów - sprzęt": data_wart_nakladow2,
        "Wartość nakładów - sprzęt informatyczny": data_wart_nakladow3,
        "Wartość nakładów - sprzęt teleinformatyczny": data_wart_nakladow4,
        "Wartość nakładów - leasing": data_wart_nakladow5,
        "Wartość nakładów - oprogramowanie": data_wart_nakladow6,
    }
    selected_page = st.sidebar.selectbox("Wybierz analizę", pages.keys())
    pages[selected_page]()

def data_wart_nakladow1():
    st.markdown('### Wartość nakładów na sprzęt i oprogramowanie')

    @st.cache_data
    def get_local_data():
        try:
            df = pd.read_excel('wart_nak_sprzet_opr_dzialy.xlsx', sheet_name=0) 
            return df.set_index('Region')
        except Exception as e:
            st.error(f'Error loading data: {e}')
            return pd.DataFrame()

    df = get_local_data()
    if df.empty:
        return

    countries = st.multiselect('Wybierz dział', list(df.index))
    if not countries:
        st.error('Wybierz co najmniej jeden.')
        return

    data = df.loc[countries].T.reset_index().rename(columns={'index': 'year'})
    data = pd.melt(data, id_vars=['year'], var_name='Region', value_name='Wartość')
    data['year'] = pd.to_numeric(data['year'], errors='coerce')
    data = data.dropna(subset=['year']).astype({'year': int})

#     sens_chart = alt.Chart(data).mark_line().encode(
#        x=alt.X('year:O',title='Rok'),y=alt.Y('Wartość:Q', title='Wartość'),color=alt.Color('Region:N',title='Branża'))
#     st.altair_chart(sens_chart, use_container_width=True)
    chart = alt.Chart(data).mark_line().encode(
        x=alt.X('year:O', title='Rok'),
        y=alt.Y('Wartość:Q', title='Wartość'),
        color=alt.Color('Region:N', title='Branża')
    ).properties(
        title=alt.TitleParams(
            text='Wartość nakładów na sprzęt i oprogramowanie w latach 2015-2022',
            anchor='middle'))
    st.altair_chart(chart, use_container_width=True)

    st.write("### Wskaźnik nachylenia Sena")
    slope_data = []
    for region in countries:
        region_data = data[data['Region'] == region]
        slope, intercept, _, _, _ = linregress(region_data['year'], region_data['Wartość'])
        slope_data.append({'Region': region, 'Slope': slope, 'Intercept': intercept})
    sens_slope_results = pd.DataFrame(slope_data)
    st.write(sens_slope_results)

    st.write("### Test Manna-Kendalla")
    mk_data = []
    for region in countries:
        region_data = data[data['Region'] == region]
        mk_result = mk.original_test(region_data['Wartość'].values)
        mk_data.append({'Region': region, 'Trend': mk_result.trend, 'p-value': mk_result.p})
    mann_kendall_results = pd.DataFrame(mk_data)
    st.write(mann_kendall_results)    
    
def data_wart_nakladow2():
    st.markdown('### Wartość nakładów na sprzęt ogółem')

    @st.cache_data
    def get_local_data():
        try:
            df = pd.read_excel('wart_nak_sprzet_dzialy.xlsx', sheet_name=0)
            return df.set_index('Region')
        except Exception as e:
            st.error(f'Error loading data: {e}')
            return pd.DataFrame()

    df = get_local_data()
    if df.empty:
        return

    countries = st.multiselect('Wybierz dział', list(df.index))
    if not countries:
        st.error('Wybierz co najmniej jeden.')
        return

    data = df.loc[countries].T.reset_index().rename(columns={'index': 'year'})
    data = pd.melt(data, id_vars=['year'], var_name='Region', value_name='Wartość')
    data['year'] = pd.to_numeric(data['year'], errors='coerce')
    data = data.dropna(subset=['year']).astype({'year': int})

#     sens_chart = alt.Chart(data).mark_line().encode(
#        x=alt.X('year:O',title='Rok'),y=alt.Y('Wartość:Q', title='Wartość'),color=alt.Color('Region:N',title='Branża'))
#     st.altair_chart(sens_chart, use_container_width=True)
    chart = alt.Chart(data).mark_line().encode(
        x=alt.X('year:O', title='Rok'),
        y=alt.Y('Wartość:Q', title='Wartość'),
        color=alt.Color('Region:N', title='Branża')
    ).properties(
        title=alt.TitleParams(
            text='Wartość nakładów na sprzęt w latach 2015-2022',
            anchor='middle'))
    st.altair_chart(chart, use_container_width=True)
    
    st.write("### Wskaźnik nachylenia Sena")
    slope_data = []
    for region in countries:
        region_data = data[data['Region'] == region]
        slope, intercept, _, _, _ = linregress(region_data['year'], region_data['Wartość'])
        slope_data.append({'Region': region, 'Slope': slope, 'Intercept': intercept})
    sens_slope_results = pd.DataFrame(slope_data)
    st.write(sens_slope_results)

    st.write("### Test Manna-Kendalla")
    mk_data = []
    for region in countries:
        region_data = data[data['Region'] == region]
        mk_result = mk.original_test(region_data['Wartość'].values)
        mk_data.append({'Region': region, 'Trend': mk_result.trend, 'p-value': mk_result.p})
    mann_kendall_results = pd.DataFrame(mk_data)
    st.write(mann_kendall_results)        

def data_wart_nakladow3():
    st.markdown('### Wartość nakładów na sprzęt informatyczny')

    @st.cache_data
    def get_local_data():
        try:
            df = pd.read_excel('wart_nak_sprzet_inf_dzialy.xlsx', sheet_name=0) 
            return df.set_index('Region')
        except Exception as e:
            st.error(f'Error loading data: {e}')
            return pd.DataFrame()

    df = get_local_data()
    if df.empty:
        return

    countries = st.multiselect('Wybierz dział', list(df.index))
    if not countries:
        st.error('Wybierz co najmniej jeden.')
        return

    data = df.loc[countries].T.reset_index().rename(columns={'index': 'year'})
    data = pd.melt(data, id_vars=['year'], var_name='Region', value_name='Wartość')
    data['year'] = pd.to_numeric(data['year'], errors='coerce')
    data = data.dropna(subset=['year']).astype({'year': int})

#     sens_chart = alt.Chart(data).mark_line().encode(
#        x=alt.X('year:O',title='Rok'),y=alt.Y('Wartość:Q', title='Wartość'),color=alt.Color('Region:N',title='Branża'))
#     st.altair_chart(sens_chart, use_container_width=True)
    chart = alt.Chart(data).mark_line().encode(
        x=alt.X('year:O', title='Rok'),
        y=alt.Y('Wartość:Q', title='Wartość'),
        color=alt.Color('Region:N', title='Branża')
    ).properties(
        title=alt.TitleParams(
            text='Wartość nakładów na sprzęt informatyczny w latach 2015-2022',
            anchor='middle'))
    st.altair_chart(chart, use_container_width=True)
    
    st.write("### Wskaźnik nachylenia Sena")
    slope_data = []
    for region in countries:
        region_data = data[data['Region'] == region]
        slope, intercept, _, _, _ = linregress(region_data['year'], region_data['Wartość'])
        slope_data.append({'Region': region, 'Slope': slope, 'Intercept': intercept})
    sens_slope_results = pd.DataFrame(slope_data)
    st.write(sens_slope_results)

    st.write("### Test Manna-Kendalla")
    mk_data = []
    for region in countries:
        region_data = data[data['Region'] == region]
        mk_result = mk.original_test(region_data['Wartość'].values)
        mk_data.append({'Region': region, 'Trend': mk_result.trend, 'p-value': mk_result.p})
    mann_kendall_results = pd.DataFrame(mk_data)
    st.write(mann_kendall_results)    
    
def data_wart_nakladow4():
    st.markdown('### Wartość nakładów na sprzęt telekomunikacyjny')

    @st.cache_data
    def get_local_data():
        try:
            df = pd.read_excel('wart_nak_sprzet_teleinf_dzialy.xlsx', sheet_name=0)
            return df.set_index('Region')
        except Exception as e:
            st.error(f'Error loading data: {e}')
            return pd.DataFrame()

    df = get_local_data()
    if df.empty:
        return

    countries = st.multiselect('Wybierz dział', list(df.index))
    if not countries:
        st.error('Wybierz co najmniej jeden.')
        return

    data = df.loc[countries].T.reset_index().rename(columns={'index': 'year'})
    data = pd.melt(data, id_vars=['year'], var_name='Region', value_name='Wartość')
    data['year'] = pd.to_numeric(data['year'], errors='coerce')
    data = data.dropna(subset=['year']).astype({'year': int})

#     sens_chart = alt.Chart(data).mark_line().encode(
#        x=alt.X('year:O',title='Rok'),y=alt.Y('Wartość:Q', title='Wartość'),color=alt.Color('Region:N',title='Branża'))
#     st.altair_chart(sens_chart, use_container_width=True)
    chart = alt.Chart(data).mark_line().encode(
        x=alt.X('year:O', title='Rok'),
        y=alt.Y('Wartość:Q', title='Wartość'),
        color=alt.Color('Region:N', title='Branża')
    ).properties(
        title=alt.TitleParams(
            text='Wartość nakładów na sprzęt telekomunikacyjny w latach 2015-2022',
            anchor='middle'))
    st.altair_chart(chart, use_container_width=True)
    
    st.write("### Wskaźnik nachylenia Sena")
    slope_data = []
    for region in countries:
        region_data = data[data['Region'] == region]
        slope, intercept, _, _, _ = linregress(region_data['year'], region_data['Wartość'])
        slope_data.append({'Region': region, 'Slope': slope, 'Intercept': intercept})
    sens_slope_results = pd.DataFrame(slope_data)
    st.write(sens_slope_results)

    st.write("### Test Manna-Kendalla")
    mk_data = []
    for region in countries:
        region_data = data[data['Region'] == region]
        mk_result = mk.original_test(region_data['Wartość'].values)
        mk_data.append({'Region': region, 'Trend': mk_result.trend, 'p-value': mk_result.p})
    mann_kendall_results = pd.DataFrame(mk_data)
    st.write(mann_kendall_results)    
    
def data_wart_nakladow5():
    st.markdown('### Wartość nakładów na leasing urządzeń ICT')

    @st.cache_data
    def get_local_data():
        try:
            df = pd.read_excel('wart_nak_leasing_dzialy.xlsx', sheet_name=0)
            return df.set_index('Region')
        except Exception as e:
            st.error(f'Error loading data: {e}')
            return pd.DataFrame()

    df = get_local_data()
    if df.empty:
        return

    countries = st.multiselect('Wybierz dział', list(df.index))
    if not countries:
        st.error('Wybierz co najmniej jeden.')
        return

    data = df.loc[countries].T.reset_index().rename(columns={'index': 'year'})
    data = pd.melt(data, id_vars=['year'], var_name='Region', value_name='Wartość')
    data['year'] = pd.to_numeric(data['year'], errors='coerce')
    data = data.dropna(subset=['year']).astype({'year': int})

#     sens_chart = alt.Chart(data).mark_line().encode(
#        x=alt.X('year:O',title='Rok'),y=alt.Y('Wartość:Q', title='Wartość'),color=alt.Color('Region:N',title='Branża'))
#     st.altair_chart(sens_chart, use_container_width=True)
    chart = alt.Chart(data).mark_line().encode(
        x=alt.X('year:O', title='Rok'),
        y=alt.Y('Wartość:Q', title='Wartość'),
        color=alt.Color('Region:N', title='Branża')
    ).properties(
        title=alt.TitleParams(
            text='Wartość nakładów na leasing urządzeń ICT w latach 2015-2022',
            anchor='middle'))
    st.altair_chart(chart, use_container_width=True)
    
    st.write("### Wskaźnik nachylenia Sena")
    slope_data = []
    for region in countries:
        region_data = data[data['Region'] == region]
        slope, intercept, _, _, _ = linregress(region_data['year'], region_data['Wartość'])
        slope_data.append({'Region': region, 'Slope': slope, 'Intercept': intercept})
    sens_slope_results = pd.DataFrame(slope_data)
    st.write(sens_slope_results)

    st.write("### Test Manna-Kendalla")
    mk_data = []
    for region in countries:
        region_data = data[data['Region'] == region]
        mk_result = mk.original_test(region_data['Wartość'].values)
        mk_data.append({'Region': region, 'Trend': mk_result.trend, 'p-value': mk_result.p})
    mann_kendall_results = pd.DataFrame(mk_data)
    st.write(mann_kendall_results)    
    
def data_wart_nakladow6():
    st.markdown('### Wartość nakładów na oprogramowanie')

    @st.cache_data
    def get_local_data():
        try:
            df = pd.read_excel('wart_nak_oprog_dzialy.xlsx', sheet_name=0)
            return df.set_index('Region')
        except Exception as e:
            st.error(f'Error loading data: {e}')
            return pd.DataFrame()

    df = get_local_data()
    if df.empty:
        return

    countries = st.multiselect('Wybierz dział', list(df.index))
    if not countries:
        st.error('Wybierz co najmniej jeden.')
        return

    data = df.loc[countries].T.reset_index().rename(columns={'index': 'year'})
    data = pd.melt(data, id_vars=['year'], var_name='Region', value_name='Wartość')
    data['year'] = pd.to_numeric(data['year'], errors='coerce')
    data = data.dropna(subset=['year']).astype({'year': int})

#     sens_chart = alt.Chart(data).mark_line().encode(
#        x=alt.X('year:O',title='Rok'),y=alt.Y('Wartość:Q', title='Wartość'),color=alt.Color('Region:N',title='Branża'))
#     st.altair_chart(sens_chart, use_container_width=True)
    chart = alt.Chart(data).mark_line().encode(
        x=alt.X('year:O', title='Rok'),
        y=alt.Y('Wartość:Q', title='Wartość'),
        color=alt.Color('Region:N', title='Branża')
    ).properties(
        title=alt.TitleParams(
            text='Wartość nakładów na oprogramowanie w latach 2015-2022',
            anchor='middle'))
    st.altair_chart(chart, use_container_width=True)
    
    st.write("### Wskaźnik nachylenia Sena")
    slope_data = []
    for region in countries:
        region_data = data[data['Region'] == region]
        slope, intercept, _, _, _ = linregress(region_data['year'], region_data['Wartość'])
        slope_data.append({'Region': region, 'Slope': slope, 'Intercept': intercept})
    sens_slope_results = pd.DataFrame(slope_data)
    st.write(sens_slope_results)

    st.write("### Test Manna-Kendalla")
    mk_data = []
    for region in countries:
        region_data = data[data['Region'] == region]
        mk_result = mk.original_test(region_data['Wartość'].values)
        mk_data.append({'Region': region, 'Trend': mk_result.trend, 'p-value': mk_result.p})
    mann_kendall_results = pd.DataFrame(mk_data)
    st.write(mann_kendall_results)    
    
def rodzaje_polaczen_section3():
    st.sidebar.header("Rodzaje Połączeń")
    pages = {
        "szerokopasmowy  dostęp do Internetu": data_rodz_pol2,
        "szerokopasmowy  dostęp do Internetu poprzez łącze DSL": data_rodz_pol3,
    }
    selected_page = st.sidebar.selectbox("Wybierz analizę", pages.keys())
    pages[selected_page]()
    
def data_rodz_pol2():
    st.markdown('### Przedsiębiorstwa z dostępem do szerokpasmowego łącza internetowego')

    @st.cache_data
    def get_local_data():
        try:
            df = pd.read_excel('rodz_pol_szer_dzialy.xlsx')
            return df.set_index('Region')
        except Exception as e:
            st.error(f'Error loading data: {e}')
            return pd.DataFrame()

    df = get_local_data()
    if df.empty:
        return

    countries = st.multiselect('Wybierz dział', list(df.index))
    if not countries:
        st.error('Wybierz co najmniej jeden.')
        return

    data = df.loc[countries].T.reset_index().rename(columns={'index': 'year'})
    data = pd.melt(data, id_vars=['year'], var_name='Region', value_name='Wartość')
    data['year'] = pd.to_numeric(data['year'], errors='coerce')
    data = data.dropna(subset=['year']).astype({'year': int})

#     sens_chart = alt.Chart(data).mark_line().encode(
#        x=alt.X('year:O',title='Rok'),y=alt.Y('Wartość:Q', title='Wartość'),color=alt.Color('Region:N',title='Branża'))
#     st.altair_chart(sens_chart, use_container_width=True)
    chart = alt.Chart(data).mark_line().encode(
        x=alt.X('year:O', title='Rok'),
        y=alt.Y('Wartość:Q', title='Wartość'),
        color=alt.Color('Region:N', title='Branża')
    ).properties(
        title=alt.TitleParams(
            text='Przedsiębiorstwa z dostępem do szerokpasmowego łącza internetowego w latach 2015-2022',
            anchor='middle',
        fontSize=12))
    st.altair_chart(chart, use_container_width=True)
    
    st.write("### Wskaźnik nachylenia Sena")
    slope_data = []
    for region in countries:
        region_data = data[data['Region'] == region]
        slope, intercept, _, _, _ = linregress(region_data['year'], region_data['Wartość'])
        slope_data.append({'Region': region, 'Slope': slope, 'Intercept': intercept})
    sens_slope_results = pd.DataFrame(slope_data)
    st.write(sens_slope_results)

    st.write("### Test Manna-Kendalla")
    mk_data = []
    for region in countries:
        region_data = data[data['Region'] == region]
        mk_result = mk.original_test(region_data['Wartość'].values)
        mk_data.append({'Region': region, 'Trend': mk_result.trend, 'p-value': mk_result.p})
    mann_kendall_results = pd.DataFrame(mk_data)
    st.write(mann_kendall_results)
    
def data_rodz_pol3():
    st.markdown('### Przedsiębiorstwa z dostępem do szerokopasmowego łącza internetowego poprzez łącze DSL')

    @st.cache_data
    def get_local_data():
        try:
            df = pd.read_excel('rodz_pol_szerDSL_dzialy.xlsx')
            return df.set_index('Region')
        except Exception as e:
            st.error(f'Error loading data: {e}')
            return pd.DataFrame()

    df = get_local_data()
    if df.empty:
        return

    countries = st.multiselect('Wybierz dział', list(df.index))
    if not countries:
        st.error('Wybierz co najmniej jeden.')
        return

    data = df.loc[countries].T.reset_index().rename(columns={'index': 'year'})
    data = pd.melt(data, id_vars=['year'], var_name='Region', value_name='Wartość')
    data['year'] = pd.to_numeric(data['year'], errors='coerce')
    data = data.dropna(subset=['year']).astype({'year': int})

#     sens_chart = alt.Chart(data).mark_line().encode(
#        x=alt.X('year:O',title='Rok'),y=alt.Y('Wartość:Q', title='Wartość'),color=alt.Color('Region:N',title='Branża'))
#     st.altair_chart(sens_chart, use_container_width=True)
    chart = alt.Chart(data).mark_line().encode(
        x=alt.X('year:O', title='Rok'),
        y=alt.Y('Wartość:Q', title='Wartość'),
        color=alt.Color('Region:N', title='Branża')
    ).properties(
        title=alt.TitleParams(
            text='Przedsiębiorstwa z dostępem do szerokopasmowego łącza internetowego poprzez łącze DSL w latach 2015-2022',
            anchor='middle',
        fontSize=12))
    st.altair_chart(chart, use_container_width=True)
    
    st.write("### Wskaźnik nachylenia Sena")
    slope_data = []
    for region in countries:
        region_data = data[data['Region'] == region]
        slope, intercept, _, _, _ = linregress(region_data['year'], region_data['Wartość'])
        slope_data.append({'Region': region, 'Slope': slope, 'Intercept': intercept})
    sens_slope_results = pd.DataFrame(slope_data)
    st.write(sens_slope_results)

    st.write("### Test Manna-Kendalla")
    mk_data = []
    for region in countries:
        region_data = data[data['Region'] == region]
        mk_result = mk.original_test(region_data['Wartość'].values)
        mk_data.append({'Region': region, 'Trend': mk_result.trend, 'p-value': mk_result.p})
    mann_kendall_results = pd.DataFrame(mk_data)
    st.write(mann_kendall_results)


if selected_section == 'Liczność firm':
    data_visualization()
elif selected_section == 'Mapy':
    mapy_section()
elif selected_section == 'Branże':
    data_dzialy()
elif selected_section == 'Korelacje-technologia':
    korelacje_tech_section()
elif selected_section == 'Korelacje-informacje':
    korelacje_infor_section()
elif selected_section == 'Korelacje-turystyka':
    korelacje_tur_section()
elif selected_section == 'Korelacje-budownictwo':
    korelacje_bud_section()
elif selected_section == 'Korelacje-przemysł':
    korelacje_przemysl_section()
elif selected_section == 'Korelacje-handel':
    korelacje_handel_section()
elif selected_section == 'Regresja liniowa':
    regresja_liniowa_section()
