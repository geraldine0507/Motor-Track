# -*- coding: utf-8 -*-
"""
Mototrack - Costos, clasificacion ABC-XYZ, motor de pronosticos e indicadores de gestion (IFR, Rotacion, GMROI).

Aplicacion local en Streamlit.
Integra datos de Mototrack con costos, precios y registros de juego (Demanda, Ventas e Inventario).
Calcula utilidad de los 8 SKU, construye ABC, ejecuta el motor de pronosticos con statsforecast,
clasifica XYZ segun el desempeno del pronostico (Score%), combina ABC-XYZ, visualiza IFR, Rotacion, GMROI
y exporta resultados completos a Excel.
"""

from __future__ import annotations

import io
import os
from datetime import datetime

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.stats import norm

from statsforecast import StatsForecast
from statsforecast.models import (
    HoltWinters,
    WindowAverage,
    SimpleExponentialSmoothingOptimized,
    MSTL,
)

import streamlit as st

# ---------------------------------------------------------------------------
# Constantes academicas
# ---------------------------------------------------------------------------

SKU_UNIVERSO = [
    ("NORTE", "MOTO"),
    ("NORTE", "CUATRIMOTO"),
    ("NORTE", "TRACTOR"),
    ("CENTRO", "MOTO"),
    ("CENTRO", "CUATRIMOTO"),
    ("CENTRO", "TRACTOR"),
    ("SUR", "MOTO"),
    ("SUR", "CUATRIMOTO"),
]

PRODUCTOS = ["MOTO", "CUATRIMOTO", "TRACTOR"]
REGIONALES = ["NORTE", "CENTRO", "SUR"]

N_WINDOWS_DEFAULT = 6
H_DEFAULT = 6
SEASON_LENGTH = 13  # 13 semanas ~ 1 trimestre
VENTANA_ANUAL_DEFAULT = 52  # 52 semanas ~ 1 ano

COLOR_PRODUCTO = {"MOTO": "salmon", "CUATRIMOTO": "navy", "TRACTOR": "darkcyan"}


# ---------------------------------------------------------------------------
# Lectura de datos (Demanda, Costos, Ventas, Inventario)
# ---------------------------------------------------------------------------

def leer_demanda(archivo) -> pd.DataFrame:
    crudo = pd.read_excel(archivo, sheet_name="Demand", header=None)
    ncols = crudo.shape[1]
    if ncols < 13:
        raise ValueError(f"La hoja 'Demand' debe tener al menos 13 columnas (A-M), tiene {ncols}.")
    if crudo.shape[0] < 3:
        raise ValueError("La hoja 'Demand' debe tener al menos 3 filas.")

    encabezados = crudo.iloc[1]

    POS_NORTE_TURN, POS_NORTE_MOTO, POS_NORTE_CUATRI, POS_NORTE_TRACTOR = 0, 1, 2, 3
    POS_CENTRO_TURN, POS_CENTRO_MOTO, POS_CENTRO_CUATRI, POS_CENTRO_TRACTOR = 5, 6, 7, 8
    POS_SUR_TURN, POS_SUR_MOTO, POS_SUR_CUATRI = 10, 11, 12

    def _check_header(idx, esperado):
        if idx >= ncols:
            raise ValueError(f"La hoja 'Demand' no tiene la columna en la posicion {idx}.")
        real = "" if pd.isna(encabezados.iloc[idx]) else str(encabezados.iloc[idx]).strip()
        if real != esperado:
            raise ValueError(f"Encabezado inesperado en posicion {idx}: esperado '{esperado}', encontrado '{real}'.")

    _check_header(POS_NORTE_TURN, "Turn")
    _check_header(POS_NORTE_MOTO, "MOTO")
    _check_header(POS_NORTE_CUATRI, "CUATRIMOTO")
    _check_header(POS_NORTE_TRACTOR, "TRACTOR")
    _check_header(POS_CENTRO_TURN, "Turn")
    _check_header(POS_CENTRO_MOTO, "MOTO")
    _check_header(POS_CENTRO_CUATRI, "CUATRIMOTO")
    _check_header(POS_CENTRO_TRACTOR, "TRACTOR")
    _check_header(POS_SUR_TURN, "Turn")
    _check_header(POS_SUR_MOTO, "MOTO")
    _check_header(POS_SUR_CUATRI, "CUATRIMOTO")

    BLOQUES = [
        ("NORTE", "MOTO", POS_NORTE_MOTO),
        ("NORTE", "CUATRIMOTO", POS_NORTE_CUATRI),
        ("NORTE", "TRACTOR", POS_NORTE_TRACTOR),
        ("CENTRO", "MOTO", POS_CENTRO_MOTO),
        ("CENTRO", "CUATRIMOTO", POS_CENTRO_CUATRI),
        ("CENTRO", "TRACTOR", POS_CENTRO_TRACTOR),
        ("SUR", "MOTO", POS_SUR_MOTO),
        ("SUR", "CUATRIMOTO", POS_SUR_CUATRI),
    ]
    POS_TURN = [POS_NORTE_TURN, POS_CENTRO_TURN, POS_SUR_TURN]

    datos = crudo.iloc[2:].reset_index(drop=True)
    registros = []
    for _, fila in datos.iterrows():
        turns_raw = [fila.iloc[idx] for idx in POS_TURN]
        if all(pd.isna(t) for t in turns_raw):
            continue
        turns_int = [int(t) for t in turns_raw if not pd.isna(t)]
        if len(set(turns_int)) > 1:
            raise ValueError(f"Los turnos de los bloques no coinciden: {turns_int}.")
        turn = turns_int[0]

        for regional, producto, pos_col in BLOQUES:
            valor = fila.iloc[pos_col]
            if pd.isna(valor):
                continue
            registros.append({
                "Turn": turn,
                "REGIONAL": regional,
                "PRODUCTO": producto,
                "DEMANDA": float(valor),
            })

    if not registros:
        raise ValueError("No se encontraron datos de demanda validos.")

    largo = pd.DataFrame(registros)
    return largo.sort_values("Turn").reset_index(drop=True)


def leer_costos(archivo) -> pd.DataFrame:
    df = pd.read_excel(archivo, sheet_name="Costos y Precios", header=None)
    filas_costos = []
    for _, fila in df.iterrows():
        bloque = None
        for i in range(len(fila) - 3):
            try:
                bloque = [float(fila[i]), float(fila[i + 1]), float(fila[i + 2]), float(fila[i + 3])]
                break
            except (ValueError, TypeError):
                continue
        if bloque is not None:
            filas_costos.append(bloque)

    if len(filas_costos) < 3:
        raise ValueError("No se pudieron identificar 3 filas de costos en 'Costos y Precios'.")

    filas_costos = filas_costos[:3]
    costos = pd.DataFrame(filas_costos, columns=["MD", "MOD", "Costo", "Precio"])
    costos.insert(0, "PRODUCTO", PRODUCTOS)
    return costos


def leer_ventas(archivo) -> pd.DataFrame:
    registros = []
    hojas_ventas = [("CENTRO", "Ventas Centro"), ("NORTE", "Ventas NORTE"), ("SUR", "Ventas SUR")]
    
    xls = pd.ExcelFile(archivo)
    for regional, sheet in hojas_ventas:
        if sheet not in xls.sheet_names:
            continue
        df = pd.read_excel(archivo, sheet_name=sheet)
        df_clean = df.iloc[1:].copy().reset_index(drop=True)
        cols = df_clean.columns
        for _, row in df_clean.iterrows():
            try:
                turn = int(row[cols[0]])
            except (ValueError, TypeError):
                continue
            
            for c_idx in range(1, len(cols)):
                prod_name = str(df.iloc[0, c_idx]).strip()
                if prod_name in PRODUCTOS:
                    val = row[cols[c_idx]]
                    if pd.notna(val):
                        registros.append({
                            "Turn": turn,
                            "REGIONAL": regional,
                            "PRODUCTO": prod_name,
                            "VENTAS": float(val)
                        })
    return pd.DataFrame(registros)


def leer_inventario(archivo) -> pd.DataFrame:
    registros = []
    hojas_inv = [("CENTRO", "INVENTARIO.CENTRO"), ("NORTE", "INVENTARIO.NORTE"), ("SUR", "INVENTARIO.SUR")]
    
    xls = pd.ExcelFile(archivo)
    for regional, sheet in hojas_inv:
        if sheet not in xls.sheet_names:
            continue
        df = pd.read_excel(archivo, sheet_name=sheet)
        df_clean = df.iloc[1:].copy().reset_index(drop=True)
        cols = df_clean.columns
        for _, row in df_clean.iterrows():
            try:
                turn = int(row[cols[0]])
            except (ValueError, TypeError):
                continue
            
            for c_idx in range(1, len(cols)):
                prod_name = str(df.iloc[0, c_idx]).strip()
                if prod_name in PRODUCTOS:
                    val = row[cols[c_idx]]
                    if pd.notna(val):
                        registros.append({
                            "Turn": turn,
                            "REGIONAL": regional,
                            "PRODUCTO": prod_name,
                            "INVENTARIO": float(val)
                        })
    return pd.DataFrame(registros)


def validar_datos(demanda: pd.DataFrame, costos: pd.DataFrame) -> dict:
    controles = []
    ok = True

    skus = set(zip(demanda["REGIONAL"], demanda["PRODUCTO"]))
    tiene_8 = skus == set(SKU_UNIVERSO)
    controles.append(("Exactamente 8 SKU (sin Tractor-Sur)", tiene_8))
    ok = ok and tiene_8

    sin_tractor_sur = ("SUR", "TRACTOR") not in skus
    controles.append(("No existe Tractor-Sur", sin_tractor_sur))
    ok = ok and sin_tractor_sur

    tres = set(costos["PRODUCTO"]) == set(PRODUCTOS)
    controles.append(("Tres productos en costos (Moto, Cuatrimoto, Tractor)", tres))
    ok = ok and tres

    costo_cuadra = np.allclose(
        costos["Costo"].astype(float),
        costos["MD"].astype(float) + costos["MOD"].astype(float),
    )
    controles.append(("Costo unitario = MD + MOD", costo_cuadra))
    ok = ok and costo_cuadra

    demanda_valida = bool(demanda["DEMANDA"].notna().all())
    controles.append(("Campos numericos de demanda validos", demanda_valida))
    ok = ok and demanda_valida

    dup = demanda.duplicated(subset=["Turn", "REGIONAL", "PRODUCTO"]).any()
    controles.append(("Sin duplicados de Turn por SKU", not bool(dup)))
    ok = ok and not bool(dup)

    turns = sorted(demanda["Turn"].unique())
    orden_ok = turns == list(range(min(turns), max(turns) + 1))
    controles.append(("Turn consecutivo (orden temporal correcto)", orden_ok))
    ok = ok and orden_ok

    return {"ok": ok, "controles": controles}


def construir_agregados(demanda: pd.DataFrame) -> pd.DataFrame:
    base = demanda.copy()
    cedi = base[base["REGIONAL"].isin(["CENTRO", "SUR"])].groupby(["Turn", "PRODUCTO"], as_index=False)["DEMANDA"].sum()
    cedi["REGIONAL"] = "CEDI"

    moto = base.groupby(["Turn", "PRODUCTO"], as_index=False)["DEMANDA"].sum()
    moto["REGIONAL"] = "MOTOTRAK"

    agregados = pd.concat([cedi, moto], ignore_index=True)
    return agregados[["Turn", "REGIONAL", "PRODUCTO", "DEMANDA"]]


# ---------------------------------------------------------------------------
# Indicadores Financieros y de Gestion de Inventarios (IFR, ROT, GMROI)
# ---------------------------------------------------------------------------

def preparar_datos_ifr(demanda: pd.DataFrame, inventario: pd.DataFrame) -> tuple:
    if demanda.empty or inventario.empty:
        return pd.DataFrame(), np.nan, {}

    df_merged = pd.merge(demanda, inventario, on=["Turn", "REGIONAL", "PRODUCTO"], how="inner")
    if df_merged.empty:
        return pd.DataFrame(), np.nan, {}

    df_merged["CUBIERTO"] = np.minimum(df_merged["DEMANDA"], df_merged["INVENTARIO"])
    df_merged["IFR"] = np.where(df_merged["DEMANDA"] > 0, df_merged["CUBIERTO"] / df_merged["DEMANDA"], 1.0)
    df_merged["REF"] = df_merged["REGIONAL"] + "-" + df_merged["PRODUCTO"]

    # IFR Global (sum total cubierto / sum total demandado)
    total_demanda = df_merged["DEMANDA"].sum()
    total_cubierto = df_merged["CUBIERTO"].sum()
    ifr_global = (total_cubierto / total_demanda) if total_demanda > 0 else np.nan

    # IFR Promedio por referencia
    ifr_por_ref = df_merged.groupby("REF")["IFR"].mean().to_dict()

    return df_merged, ifr_global, ifr_por_ref


def calcular_rotacion_gmroi(ventas: pd.DataFrame, inventario: pd.DataFrame, costos: pd.DataFrame) -> pd.DataFrame:
    if ventas.empty or inventario.empty:
        return pd.DataFrame()

    v_sum = ventas.groupby(["REGIONAL", "PRODUCTO"], as_index=False)["VENTAS"].sum()
    i_avg = inventario.groupby(["REGIONAL", "PRODUCTO"], as_index=False)["INVENTARIO"].mean()

    res = pd.merge(v_sum, i_avg, on=["REGIONAL", "PRODUCTO"], how="outer").fillna(0.0)
    res = pd.merge(res, costos[["PRODUCTO", "Costo", "Precio"]], on="PRODUCTO", how="left")

    res["ROT"] = np.where(res["INVENTARIO"] > 0, res["VENTAS"] / res["INVENTARIO"], np.nan)
    res["Utilidad unitaria"] = res["Precio"] - res["Costo"]
    res["Utilidad total"] = res["VENTAS"] * res["Utilidad unitaria"]
    res["Valor inventario promedio"] = res["INVENTARIO"] * res["Costo"]
    res["GMROI"] = np.where(res["Valor inventario promedio"] > 0, res["Utilidad total"] / res["Valor inventario promedio"], np.nan)
    res["REF"] = res["REGIONAL"] + "-" + res["PRODUCTO"]

    return res


# ---------------------------------------------------------------------------
# Graficas de Indicadores (IFR, Rotacion, GMROI)
# ---------------------------------------------------------------------------

def grafica_ifr_semanal_por_referencia(ifr_df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    colores = {"NORTE": "#e74c3c", "CENTRO": "#3498db", "SUR": "#2ecc71"}

    for ref in sorted(ifr_df["REF"].unique()):
        sub = ifr_df[ifr_df["REF"] == ref].sort_values("Turn")
        reg = sub["REGIONAL"].iloc[0]
        fig.add_trace(go.Scatter(
            x=sub["Turn"],
            y=sub["IFR"] * 100,
            mode="lines+markers",
            name=ref,
            line=dict(width=2),
        ))

    fig.update_layout(
        title="Item Fill Rate (IFR %) Semanal por Referencia",
        xaxis_title="Turno / Semana",
        yaxis_title="IFR (%)",
        template="plotly_white",
        height=450,
        hovermode="x unified",
    )
    return fig


def grafica_ifr_global(ifr_global: float, ifr_por_ref: dict) -> go.Figure:
    refs = list(ifr_por_ref.keys())
    valores = [ifr_por_ref[r] * 100 for r in refs]
    etiquetas = list(refs)

    if not pd.isna(ifr_global):
        valores.append(ifr_global * 100)
        etiquetas.append("GLOBAL")

    fig = go.Figure()
    colors = ["#3498db"] * len(refs) + ["#2ecc71"]
    fig.add_trace(go.Bar(
        x=etiquetas,
        y=valores,
        marker_color=colors,
        text=[f"{v:.1f}%" for v in valores],
        textposition="auto",
    ))

    fig.update_layout(
        title="IFR Promedio por Referencia vs. IFR Global",
        yaxis_title="IFR (%)",
        template="plotly_white",
        height=400,
    )
    return fig


def grafica_rotacion_inventarios(rot_df: pd.DataFrame) -> go.Figure:
    df_sorted = rot_df.sort_values("ROT", ascending=False)
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=df_sorted["REF"],
        y=df_sorted["ROT"],
        marker_color="#2980b9",
        text=df_sorted["ROT"].apply(lambda x: f"{x:.2f}x" if pd.notna(x) else "N/A"),
        textposition="auto",
    ))
    fig.update_layout(
        title="Rotación de Inventarios (ROT = Ventas Totales / Inventario Promedio)",
        yaxis_title="Veces (x)",
        template="plotly_white",
        height=400,
    )
    return fig


def grafica_gmroi(gmroi_df: pd.DataFrame) -> go.Figure:
    df_sorted = gmroi_df.sort_values("GMROI", ascending=False)
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=df_sorted["REF"],
        y=df_sorted["GMROI"],
        marker_color="#27ae60",
        text=df_sorted["GMROI"].apply(lambda x: f"{x:.2f}x" if pd.notna(x) else "N/A"),
        textposition="auto",
    ))
    fig.update_layout(
        title="GMROI (Utilidad Bruta Total / Valor Inventario Promedio)",
        yaxis_title="Retorno ($ Utilidad / $ Inventario)",
        template="plotly_white",
        height=400,
    )
    return fig


# ---------------------------------------------------------------------------
# ABC, Pronosticos y Revision Periodica
# ---------------------------------------------------------------------------

def calcular_abc(demanda: pd.DataFrame, costos: pd.DataFrame, ventana: int = VENTANA_ANUAL_DEFAULT) -> tuple:
    turn_max = int(demanda["Turn"].max())
    turn_min_ventana = max(int(demanda["Turn"].min()), turn_max - ventana + 1)
    dem_ventana = demanda[demanda["Turn"] >= turn_min_ventana].copy()

    info_ventana = {
        "turn_inicial": turn_min_ventana,
        "turn_final": turn_max,
        "observaciones": int(dem_ventana["Turn"].nunique()),
    }

    dem_anual = dem_ventana.groupby(["REGIONAL", "PRODUCTO"], as_index=False)["DEMANDA"].sum().rename(columns={"DEMANDA": "Demanda anual"})
    abc = dem_anual.merge(costos, on="PRODUCTO", how="left")
    abc["Utilidad unitaria"] = abc["Precio"].astype(float) - abc["Costo"].astype(float)
    abc["Utilidad total"] = abc["Demanda anual"] * abc["Utilidad unitaria"]
    abc = abc.sort_values("Utilidad total", ascending=False).reset_index(drop=True)

    total_portafolio = abc["Utilidad total"].sum()
    abc["Participacion"] = abc["Utilidad total"] / total_portafolio
    abc["Acumulado"] = abc["Participacion"].cumsum()

    def _clase(acum):
        if acum <= 0.80:
            return "A"
        elif acum <= 0.95:
            return "B"
        return "C"

    abc["ABC"] = abc["Acumulado"].apply(_clase)
    abc.insert(0, "Ranking", abc.index + 1)
    abc["SKU"] = abc["PRODUCTO"] + "-" + abc["REGIONAL"]

    columnas = [
        "Ranking", "SKU", "REGIONAL", "PRODUCTO", "Demanda anual",
        "MD", "MOD", "Costo", "Precio", "Utilidad unitaria",
        "Utilidad total", "Participacion", "Acumulado", "ABC",
    ]
    return abc[columnas], info_ventana


def preparar_series(demanda: pd.DataFrame) -> pd.DataFrame:
    df = demanda.copy()
    df["unique_id"] = df["REGIONAL"] + "|" + df["PRODUCTO"]
    df["ds"] = pd.to_datetime("2020-01-06") + pd.to_timedelta(df["Turn"] - 1, unit="W")
    df = df.rename(columns={"DEMANDA": "y"})
    return df[["unique_id", "ds", "y"]].sort_values(["unique_id", "ds"])


def obtener_modelos():
    return [
        HoltWinters(season_length=SEASON_LENGTH, alias="hw"),
        WindowAverage(window_size=3, alias="wa_3"),
        WindowAverage(window_size=6, alias="wa_6"),
        WindowAverage(window_size=12, alias="wa_12"),
        SimpleExponentialSmoothingOptimized(alias="ses"),
        MSTL(season_length=SEASON_LENGTH, alias="mstl"),
    ]


def calcular_metricas(y_real: np.ndarray, y_pred: np.ndarray) -> dict:
    y_real = np.asarray(y_real, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    suma_y = np.sum(y_real)
    if suma_y == 0:
        return {"MAE%": np.nan, "Sesgo%": np.nan, "Score%": np.nan, "RMSE": np.nan}

    mae = np.sum(np.abs(y_real - y_pred))
    sesgo = np.sum(y_real - y_pred)
    mae_pct = mae / suma_y
    sesgo_pct = sesgo / suma_y
    score_pct = mae_pct + abs(sesgo_pct)
    rmse = float(np.sqrt(np.mean((y_real - y_pred) ** 2)))
    return {"MAE%": mae_pct, "Sesgo%": sesgo_pct, "Score%": score_pct, "RMSE": rmse}


def ejecutar_backtesting(df_ts: pd.DataFrame, n_windows: int, h: int) -> tuple:
    modelos = obtener_modelos()
    alias_modelos = [m.alias for m in modelos]
    series_ids = list(df_ts["unique_id"].unique())

    comparacion_registros, ganadores, pronosticos, errores = [], {}, {}, []

    for uid in series_ids:
        sub = df_ts[df_ts["unique_id"] == uid][["unique_id", "ds", "y"]].copy()
        metricas_serie = {}
        try:
            sf = StatsForecast(models=modelos, freq="W-MON", n_jobs=-1)
            cv = sf.cross_validation(df=sub, h=h, step_size=h, n_windows=n_windows)
            for alias in alias_modelos:
                if alias not in cv.columns:
                    continue
                mask = cv[alias].notna()
                if not mask.any():
                    continue
                m = calcular_metricas(cv["y"].values[mask.values], cv[alias].values[mask.values])
                metricas_serie[alias] = m
                comparacion_registros.append({"unique_id": uid, "MODELO": alias, **m})
        except Exception as e:
            errores.append(f"{uid}: error en backtesting -> {e}")

        if metricas_serie:
            ganador = min(metricas_serie.items(), key=lambda kv: (kv[1]["Score%"], kv[1]["RMSE"]))
            ganadores[uid] = {"MODELO": ganador[0], **ganador[1]}
            try:
                m_g = next(m for m in modelos if m.alias == ganador[0])
                fc = StatsForecast(models=[m_g], freq="W-MON", n_jobs=-1).forecast(df=sub, h=h).reset_index().rename(columns={ganador[0]: "yhat"})
                pronosticos[uid] = fc[["ds", "yhat"]]
            except Exception as e:
                errores.append(f"{uid}: error al pronosticar -> {e}")

    return pd.DataFrame(comparacion_registros), ganadores, pronosticos, errores


def clasificar_xyz(ganadores: dict, limite_xy: float, limite_yz: float) -> dict:
    if limite_xy >= limite_yz:
        raise ValueError("Umbrales XYZ inconsistentes: limite X/Y debe ser < limite Y/Z.")

    xyz = {}
    for uid, info in ganadores.items():
        score = info.get("Score%", np.nan)
        if pd.isna(score):
            xyz[uid] = None
            continue
        xyz[uid] = "X" if score <= limite_xy else ("Y" if score <= limite_yz else "Z")
    return xyz


def combinar_abc_xyz(abc: pd.DataFrame, ganadores: dict, xyz: dict) -> pd.DataFrame:
    filas = []
    for _, r in abc.iterrows():
        uid = f"{r['REGIONAL']}|{r['PRODUCTO']}"
        g = ganadores.get(uid, {})
        clase_xyz = xyz.get(uid)
        filas.append({
            "Ranking": r["Ranking"],
            "SKU": r["SKU"],
            "Regional": r["REGIONAL"],
            "Producto": r["PRODUCTO"],
            "Demanda anual": r["Demanda anual"],
            "MD": r["MD"],
            "MOD": r["MOD"],
            "Costo unitario": r["Costo"],
            "Precio": r["Precio"],
            "Utilidad unitaria": r["Utilidad unitaria"],
            "Utilidad total": r["Utilidad total"],
            "Participacion %": r["Participacion"] * 100,
            "Acumulado %": r["Acumulado"] * 100,
            "ABC": r["ABC"],
            "Modelo ganador": g.get("MODELO", "N/A"),
            "MAE%": g.get("MAE%", np.nan),
            "Sesgo%": g.get("Sesgo%", np.nan),
            "Score%": g.get("Score%", np.nan),
            "RMSE": g.get("RMSE", np.nan),
            "XYZ": clase_xyz if clase_xyz else "?",
            "ABC-XYZ": f"{r['ABC']}{clase_xyz}" if clase_xyz else f"{r['ABC']}?",
        })
    return pd.DataFrame(filas)


def consolidado_producto(demanda: pd.DataFrame, costos: pd.DataFrame, ventana: int) -> pd.DataFrame:
    turn_max = int(demanda["Turn"].max())
    turn_min = max(int(demanda["Turn"].min()), turn_max - ventana + 1)
    dem = demanda[demanda["Turn"] >= turn_min]

    dem_prod = dem.groupby("PRODUCTO", as_index=False)["DEMANDA"].sum().rename(columns={"DEMANDA": "Demanda anual"})
    cons = dem_prod.merge(costos, on="PRODUCTO", how="left")
    cons["Utilidad unitaria"] = cons["Precio"].astype(float) - cons["Costo"].astype(float)
    cons["Utilidad total"] = cons["Demanda anual"] * cons["Utilidad unitaria"]
    cons["Participacion %"] = (cons["Utilidad total"] / cons["Utilidad total"].sum()) * 100
    return cons.sort_values("Utilidad total", ascending=False).reset_index(drop=True)


def disenar_revision_periodica(ganadores: dict, pronosticos: dict, R: int, L: int, Z: float) -> pd.DataFrame:
    import math
    P = int(R) + int(L)
    filas = []
    for uid, info in ganadores.items():
        regional, producto = uid.split("|")
        if regional not in {"NORTE", "CENTRO", "SUR", "CEDI"}:
            continue
        rmse = info.get("RMSE", np.nan)
        yhat = pronosticos.get(uid, pd.DataFrame()).get("yhat", np.array([]))

        demanda_p = float(np.sum(yhat[:P])) if P > 0 and len(yhat) >= P else np.nan
        sigma_p = float(rmse) * math.sqrt(P) if not pd.isna(rmse) and P > 0 else np.nan
        ss = Z * sigma_p if not pd.isna(sigma_p) else np.nan

        filas.append({
            "Regional": regional,
            "Producto": producto,
            "Modelo ganador": info.get("MODELO", "N/A"),
            "R (revision)": int(R),
            "L (entrega)": int(L),
            "P = R+L": P,
            "Demanda en P (suma pronosticos)": demanda_p,
            "RMSE (motor)": rmse,
            "sigma_P = RMSE*sqrt(P)": sigma_p,
            "Stock de seguridad": ss,
            "Nivel objetivo S": (demanda_p + ss) if not pd.isna(demanda_p) and not pd.isna(ss) else np.nan,
            "Cubre horizonte": P > 0 and len(yhat) >= P,
        })
    df = pd.DataFrame(filas)
    if not df.empty:
        df["_o"] = df["Regional"].map({"NORTE": 0, "CENTRO": 1, "SUR": 2, "CEDI": 3}).fillna(9)
        df = df.sort_values(["_o", "Producto"]).drop(columns="_o").reset_index(drop=True)
    return df


# ---------------------------------------------------------------------------
# Exportacion a Excel
# ---------------------------------------------------------------------------

def generar_excel(abc: pd.DataFrame, costos: pd.DataFrame, abc_xyz: pd.DataFrame,
                  comparacion: pd.DataFrame, pronosticos: dict,
                  info_ventana: dict, n_windows: int, h: int,
                  limite_xy: float, limite_yz: float,
                  conclusiones: list, ruta_salida: str,
                  rs_tabla: pd.DataFrame = None,
                  demanda: pd.DataFrame = None,
                  ventas: pd.DataFrame = None,
                  inventario: pd.DataFrame = None) -> None:

    utilidad_total = float(abc["Utilidad total"].sum())
    conteos_abc = abc["ABC"].value_counts().to_dict()
    conteos_xyz = (abc_xyz["XYZ"].value_counts().to_dict() if not abc_xyz.empty else {})

    resumen = pd.DataFrame({
        "Campo": [
            "Fecha/hora", "Archivo fuente", "Ventana anual (Turn inicial)",
            "Ventana anual (Turn final)", "Observaciones en ventana",
            "n_windows", "h", "Umbral X/Y (Score%)", "Umbral Y/Z (Score%)",
            "Utilidad total portafolio", "Conteo A", "Conteo B", "Conteo C",
            "Conteo X", "Conteo Y", "Conteo Z",
        ],
        "Valor": [
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            st.session_state.get("nombre_archivo", "N/A"),
            info_ventana["turn_inicial"], info_ventana["turn_final"],
            info_ventana["observaciones"], n_windows, h,
            limite_xy, limite_yz, utilidad_total,
            conteos_abc.get("A", 0), conteos_abc.get("B", 0),
            conteos_abc.get("C", 0), conteos_xyz.get("X", 0),
            conteos_xyz.get("Y", 0), conteos_xyz.get("Z", 0),
        ],
    })

    turn_max = info_ventana["turn_final"]
    pron_rows = []
    for uid, fc in pronosticos.items():
        regional, producto = uid.split("|")
        for i, (_, row) in enumerate(fc.iterrows(), start=1):
            pron_rows.append({
                "REGIONAL": regional, "PRODUCTO": producto,
                "MODELO": "ganador",
                "Turn futuro": turn_max + i,
                "Pronostico": float(row["yhat"]),
            })
    pronosticos_df = pd.DataFrame(pron_rows)
    concl_df = pd.DataFrame(conclusiones, columns=["Hallazgo / Observacion"])

    with pd.ExcelWriter(ruta_salida, engine="openpyxl") as writer:
        resumen.to_excel(writer, sheet_name="Resumen", index=False)
        costos.to_excel(writer, sheet_name="Costos", index=False)
        abc_xyz.to_excel(writer, sheet_name="Clasif ABC_XYZ", index=False)
        pronosticos_df.to_excel(writer, sheet_name="Pronosticos", index=False)
        comparacion.to_excel(writer, sheet_name="Comparacion_Modelos", index=False)

        if demanda is not None and inventario is not None:
            ifr_df, ifr_global, ifr_por_ref = preparar_datos_ifr(demanda, inventario)
            if not ifr_df.empty:
                ifr_df[["Turn", "REGIONAL", "PRODUCTO", "DEMANDA", "INVENTARIO", "IFR"]].to_excel(writer, sheet_name="IFR_Semanal", index=False)
                df_ifr_res = pd.DataFrame(list(ifr_por_ref.items()), columns=["REF", "IFR Promedio"])
                df_ifr_res.loc[len(df_ifr_res)] = ["GLOBAL", ifr_global]
                df_ifr_res.to_excel(writer, sheet_name="IFR_Resumen", index=False)

        if ventas is not None and inventario is not None:
            rot_gmroi_df = calcular_rotacion_gmroi(ventas, inventario, costos)
            if not rot_gmroi_df.empty:
                rot_gmroi_df[["REF", "REGIONAL", "PRODUCTO", "VENTAS", "INVENTARIO", "ROT", "GMROI"]].to_excel(writer, sheet_name="Rotacion_y_GMROI", index=False)

        if rs_tabla is not None and not rs_tabla.empty:
            rs_tabla.to_excel(writer, sheet_name="Revision_Periodica_RS", index=False)

        concl_df.to_excel(writer, sheet_name="Conclusiones", index=False)


# ---------------------------------------------------------------------------
# Visualizaciones Oficiales
# ---------------------------------------------------------------------------

def grafica_pronosticos_oficial(df_ts: pd.DataFrame, pronosticos: dict, h: int, turn_max: int) -> go.Figure:
    titulos = ["NORTE", "CENTRO", "SUR", "CEDI (C+S)", "MOTOTRAK (N+C+S)", ""]
    fig = make_subplots(rows=2, cols=3, subplot_titles=titulos)

    paneles = {
        (1, 1): ["NORTE|MOTO", "NORTE|CUATRIMOTO", "NORTE|TRACTOR"],
        (1, 2): ["CENTRO|MOTO", "CENTRO|CUATRIMOTO", "CENTRO|TRACTOR"],
        (1, 3): ["SUR|MOTO", "SUR|CUATRIMOTO"],
        (2, 1): ["CEDI|MOTO", "CEDI|CUATRIMOTO", "CEDI|TRACTOR"],
        (2, 2): ["MOTOTRAK|MOTO", "MOTOTRAK|CUATRIMOTO", "MOTOTRAK|TRACTOR"],
    }

    ventana_inicio = turn_max - (VENTANA_ANUAL_DEFAULT + h) + 1
    fecha_corte = pd.to_datetime("2020-01-06") + pd.to_timedelta(ventana_inicio - 1, unit="W")
    leyenda_agregada = set()

    for (row, col), uids in paneles.items():
        for uid in uids:
            producto = uid.split("|")[1]
            color = COLOR_PRODUCTO[producto]
            show = producto not in leyenda_agregada
            leyenda_agregada.add(producto)

            d = df_ts[(df_ts["unique_id"] == uid) & (df_ts["ds"] >= fecha_corte)]
            if not d.empty:
                fig.add_trace(go.Scatter(x=d["ds"], y=d["y"], mode="lines", line=dict(color=color, width=2), name=producto, legendgroup=producto, showlegend=show), row=row, col=col)

            if uid in pronosticos:
                fc = pronosticos[uid]
                fig.add_trace(go.Scatter(x=fc["ds"], y=fc["yhat"], mode="lines", line=dict(color=color, width=2, dash="dash"), name=producto, legendgroup=producto, showlegend=False), row=row, col=col)

    fig.update_layout(template="ggplot2", height=650, title="Pronósticos por región y agregados (real vs modelo ganador)")
    return fig


def grafica_pareto(abc: pd.DataFrame) -> go.Figure:
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Bar(x=abc["SKU"], y=abc["Participacion"] * 100, name="Participacion %", marker_color="steelblue"), secondary_y=False)
    fig.add_trace(go.Scatter(x=abc["SKU"], y=abc["Acumulado"] * 100, name="Acumulado %", mode="lines+markers", line=dict(color="firebrick")), secondary_y=True)
    fig.update_layout(template="ggplot2", title="Pareto ABC (por Utilidad total)", height=400)
    return fig


# ---------------------------------------------------------------------------
# Interfaz Streamlit
# ---------------------------------------------------------------------------

@st.cache_data(show_spinner="Ejecutando motor de pronósticos...")
def _motor_cacheado(demanda: pd.DataFrame, n_windows: int, h: int):
    agregados = construir_agregados(demanda)
    df_ts = preparar_series(pd.concat([demanda, agregados], ignore_index=True))
    comparacion, ganadores, pronosticos, errores = ejecutar_backtesting(df_ts, n_windows=n_windows, h=h)
    return comparacion, ganadores, pronosticos, errores, df_ts


def main():
    st.set_page_config(page_title="Mototrack - Gestión de Inventarios", layout="wide")
    st.title("Mototrack - Análisis de Inventarios y Operaciones")

    st.sidebar.header("1. Carga de datos")
    archivo = st.sidebar.file_uploader("Cargar Excel de Mototrack (.xlsx)", type=["xlsx"])

    if archivo is None:
        st.info("Cargue el archivo Excel de Mototrack para comenzar.")
        return

    st.session_state["nombre_archivo"] = archivo.name

    try:
        demanda = leer_demanda(archivo)
        costos = leer_costos(archivo)
        ventas = leer_ventas(archivo)
        inventario = leer_inventario(archivo)
    except Exception as e:
        st.error(f"Error al leer el archivo: {e}")
        return

    st.sidebar.header("Parámetros")
    ventana = st.sidebar.number_input("Ventana anual (turnos)", min_value=13, max_value=200, value=VENTANA_ANUAL_DEFAULT, step=1)
    n_windows = st.sidebar.number_input("n_windows (validación cruzada)", min_value=1, max_value=20, value=N_WINDOWS_DEFAULT, step=1)
    h = st.sidebar.number_input("h (periodos a pronosticar)", min_value=1, max_value=24, value=H_DEFAULT, step=1)

    validacion = validar_datos(demanda, costos)
    turn_max = int(demanda["Turn"].max())

    if not validacion["ok"]:
        st.error("Error en validación de datos:")
        for desc, estado in validacion["controles"]:
            st.write(f"{'OK' if estado else 'FALLA'}: {desc}")
        return

    abc, info_ventana = calcular_abc(demanda, costos, ventana)
    comparacion, ganadores, pronosticos, errores, df_ts = _motor_cacheado(demanda, n_windows, h)

    scores_validos = [v.get("Score%", np.nan) for v in ganadores.values() if not pd.isna(v.get("Score%", np.nan))]
    default_xy = float(np.percentile(scores_validos, 33)) if scores_validos else 0.30
    default_yz = float(np.percentile(scores_validos, 67)) if scores_validos else 0.60

    tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9, tab10 = st.tabs([
        "1. Carga y Validación",
        "2. Costos y Precios",
        "3. Clasificación ABC",
        "4. Pronósticos",
        "5. Clasificación XYZ",
        "6. Matriz ABC-XYZ",
        "7. Indicadores IFR",
        "8. Rotación y GMROI",
        "9. Revisión Periódica (R,S)",
        "10. Exportación y Conclusiones",
    ])

    with tab1:
        st.success("Validación de datos correcta.")
        for desc, estado in validacion["controles"]:
            st.write(f"✔️ {desc}" if estado else f"❌ {desc}")

    with tab2:
        st.dataframe(costos, use_container_width=True)

    with tab3:
        st.dataframe(abc, use_container_width=True)
        st.plotly_chart(grafica_pareto(abc), use_container_width=True)

    with tab4:
        st.dataframe(comparacion, use_container_width=True)
        st.plotly_chart(grafica_pronosticos_oficial(df_ts, pronosticos, h, turn_max), use_container_width=True)

    with tab5:
        col1, col2 = st.columns(2)
        limite_xy = col1.number_input("Límite X/Y (Score%)", 0.0, 1.0, float(default_xy), 0.01)
        limite_yz = col2.number_input("Límite Y/Z (Score%)", 0.0, 1.0, float(default_yz), 0.01)

    umbral_ok = limite_xy < limite_yz
    xyz = clasificar_xyz(ganadores, limite_xy, limite_yz) if umbral_ok else {}

    with tab6:
        if umbral_ok:
            abc_xyz = combinar_abc_xyz(abc, ganadores, xyz)
            st.dataframe(abc_xyz, use_container_width=True)
        else:
            abc_xyz = pd.DataFrame()

    with tab7:
        st.subheader("Tasa de Llenado (Item Fill Rate – IFR)")
        st.markdown(
            "El **IFR** mide el porcentaje de la demanda que fue satisfecha directamente desde el inventario disponible en cada semana ($IFR = \\min(Inventario, Demanda) / Demanda$)."
        )
        ifr_df, ifr_global, ifr_por_ref = preparar_datos_ifr(demanda, inventario)

        if not ifr_df.empty:
            col_g1, col_g2 = st.columns([1, 3])
            col_g1.metric("IFR Global del Portafolio", f"{ifr_global * 100:.2f}%" if pd.notna(ifr_global) else "N/A")
            col_g2.caption("Este indicador global sintetiza el nivel de servicio acumulado para todas las referencias y regiones en una sola métrica.")

            st.plotly_chart(grafica_ifr_semanal_por_referencia(ifr_df), use_container_width=True)
            st.plotly_chart(grafica_ifr_global(ifr_global, ifr_por_ref), use_container_width=True)
            
            with st.expander("Ver detalle de datos IFR por semana"):
                st.dataframe(ifr_df[["Turn", "REGIONAL", "PRODUCTO", "DEMANDA", "INVENTARIO", "IFR"]], use_container_width=True)
        else:
            st.warning("No se encontraron registros suficientes para calcular el IFR.")

    with tab8:
        st.subheader("Rotación de Inventarios y GMROI")
        st.markdown(
            "**Rotación de Inventarios (ROT):** Mide la velocidad con la que gira el stock ($Ventas / Inventario\\_Promedio$).\n\n"
            "**GMROI (Gross Margin Return on Investment):** Mide el retorno en utilidad bruta generado por cada peso invertido en inventario promedio."
        )
        rot_gmroi_df = calcular_rotacion_gmroi(ventas, inventario, costos)

        if not rot_gmroi_df.empty:
            st.plotly_chart(grafica_rotacion_inventarios(rot_gmroi_df), use_container_width=True)
            st.plotly_chart(grafica_gmroi(rot_gmroi_df), use_container_width=True)

            st.subheader("Tabla Comparativa de Desempeño Financiero y Operativo")
            st.dataframe(
                rot_gmroi_df[["REF", "REGIONAL", "PRODUCTO", "VENTAS", "INVENTARIO", "ROT", "GMROI"]],
                use_container_width=True
            )
        else:
            st.warning("No se encontraron datos de ventas e inventarios para calcular la Rotación y GMROI.")

    with tab9:
        st.subheader("Modelo de Inventarios de Revisión Periódica (R, S)")
        rs1, rs2, rs3 = st.columns(3)
        R = rs1.number_input("R (Periodo revisión)", 1, 20, 4)
        L = rs2.number_input("L (Lead time)", 0, 20, 2)
        NC = rs3.number_input("Nivel de Servicio (%)", 50.0, 99.99, 95.0)
        Z = float(norm.ppf(NC / 100.0))

        rs_tabla = disenar_revision_periodica(ganadores, pronosticos, R, L, Z)
        st.dataframe(rs_tabla, use_container_width=True)

    with tab10:
        st.subheader("Conclusiones y Exportación de Resultados")
        if "conclusiones" not in st.session_state:
            st.session_state["conclusiones"] = ["", "", "", ""]

        for i in range(len(st.session_state["conclusiones"])):
            st.session_state["conclusiones"][i] = st.text_area(f"Conclusión {i+1}", value=st.session_state["conclusiones"][i], key=f"c_{i}", height=70)

        nombre_salida = "Resultados_Mototrack_Completo.xlsx"
        ruta_salida = os.path.join(os.getcwd(), nombre_salida)

        if st.button("Generar y Exportar Excel de Resultados"):
            try:
                generar_excel(
                    abc, costos, abc_xyz, comparacion, pronosticos,
                    info_ventana, n_windows, h, limite_xy, limite_yz,
                    [c for c in st.session_state["conclusiones"] if c.strip()],
                    ruta_salida,
                    rs_tabla=rs_tabla,
                    demanda=demanda,
                    ventas=ventas,
                    inventario=inventario
                )
                st.success(f"Archivo generado con éxito: {nombre_salida}")
                with open(ruta_salida, "rb") as f:
                    st.download_button("Descargar Excel", data=f.read(), file_name=nombre_salida, mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
            except Exception as e:
                st.error(f"Error al generar Excel: {e}")


if __name__ == "__main__":
    main()









