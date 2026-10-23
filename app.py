# -*- coding: utf-8 -*-
"""
Mototrack - Costos, clasificacion ABC-XYZ y motor de pronosticos.

Aplicacion local en Streamlit (alternativa al desarrollo manual en Excel).
Integra datos de Mototrack con costos y precios validados, calcula utilidad
de los 8 SKU, construye ABC, ejecuta el motor de pronosticos con
statsforecast, clasifica XYZ segun el desempeno del pronostico (Score%),
combina ABC-XYZ, visualiza y exporta resultados a Excel.

Reglas academicas (CLAUDE.md):
  - Costo unitario de fabricacion = MD + MOD (sin transporte/almacenamiento).
  - Utilidad unitaria = Precio de venta - Costo unitario.
  - Utilidad total SKU = Demanda del ultimo ano x Utilidad unitaria.
  - ABC se construye con utilidad total (no forzar Pareto 80/20).
  - Universo ABC-XYZ = exactamente 8 SKU (sin Tractor-Sur).
  - XYZ usa Score% del modelo ganador (no CV).
  - Modelo ganador = menor Score%.
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
# Constantes academicas (no negociables segun CLAUDE.md secciones 3 y 4)
# ---------------------------------------------------------------------------

# Universo de 8 SKU: (REGIONAL, PRODUCTO). NO existe Tractor-Sur.
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

# Parametros oficiales del motor de pronosticos (seccion 10).
N_WINDOWS_DEFAULT = 6
H_DEFAULT = 6
SEASON_LENGTH = 13  # 13 semanas ~ 1 trimestre (estacionalidad oficial).

# Ventana anual movil para ABC (seccion 8). 52 semanas ~ 1 ano.
VENTANA_ANUAL_DEFAULT = 52

# Colores oficiales de visualizacion (seccion 19).
COLOR_PRODUCTO = {"MOTO": "salmon", "CUATRIMOTO": "navy", "TRACTOR": "darkcyan"}


# ---------------------------------------------------------------------------
# Lectura de datos (seccion 5)
# ---------------------------------------------------------------------------

def leer_demanda(archivo) -> pd.DataFrame:
    """Lee la hoja 'Demand' del Excel de Mototrack.

    Estructura oficial (seccion 5.1):
      - fila 1: nombres decorativos de tiendas.
      - fila 2: encabezados reales (se validan por posicion, sin pasarlos
        por pandas para no deduplicar los nombres repetidos).
      - fila 3 en adelante: datos.
      - Norte: col A=Turn, B=MOTO, C=CUATRIMOTO, D=TRACTOR.
      - Centro: col F=Turn, G=MOTO, H=CUATRIMOTO, I=TRACTOR.
      - Sur: col K=Turn, L=MOTO, M=CUATRIMOTO (sin Tractor-Sur).
      - E y J: separadores vacios.
      - 'Turn' es entero consecutivo; un turno = un periodo.

    Devuelve un DataFrame largo: Turn | REGIONAL | PRODUCTO | DEMANDA
    """
    # Se lee SIN encabezados (header=None) porque la hoja repite los nombres
    # "Turn", "MOTO", "CUATRIMOTO" y "TRACTOR" en los tres bloques
    # Norte/Centro/Sur. Si se usara header=1, pandas renombraria las
    # repeticiones a "Turn.1", "MOTO.2", etc. y la validacion por posicion
    # fallaria contra esos nombres artificialmente deduplicados. Con
    # header=None se validan los valores crudos de la fila 2 (indice 1).
    crudo = pd.read_excel(archivo, sheet_name="Demand", header=None)

    ncols = crudo.shape[1]
    if ncols < 13:
        raise ValueError(
            f"La hoja 'Demand' debe tener al menos 13 columnas (A-M), "
            f"tiene {ncols}. Estructura inesperada."
        )

    if crudo.shape[0] < 3:
        raise ValueError(
            "La hoja 'Demand' debe tener al menos 3 filas "
            "(fila decorativa, fila de encabezados y datos). "
            "Estructura inesperada."
        )

    # Fila 2 (indice 1) contiene los encabezados reales.
    encabezados = crudo.iloc[1]

    # Mapeo explícito por POSICION de columna (índice base 0 del DataFrame).
    # A=0, B=1, C=2, D=3, E=4(sep), F=5, G=6, H=7, I=8, J=9(sep), K=10, L=11, M=12
    POS_NORTE_TURN = 0
    POS_NORTE_MOTO = 1
    POS_NORTE_CUATRI = 2
    POS_NORTE_TRACTOR = 3
    POS_CENTRO_TURN = 5
    POS_CENTRO_MOTO = 6
    POS_CENTRO_CUATRI = 7
    POS_CENTRO_TRACTOR = 8
    POS_SUR_TURN = 10
    POS_SUR_MOTO = 11
    POS_SUR_CUATRI = 12

    # Validar encabezados en las posiciones esperadas.
    def _check_header(idx, esperado):
        if idx >= ncols:
            raise ValueError(
                f"La hoja 'Demand' no tiene la columna en la posicion {idx} "
                f"(esperado '{esperado}'). Estructura inesperada."
            )
        real = "" if pd.isna(encabezados.iloc[idx]) else str(encabezados.iloc[idx]).strip()
        if real != esperado:
            raise ValueError(
                f"Encabezado inesperado en posicion {idx}: se esperaba "
                f"'{esperado}', se encontro '{real}'. "
                "La estructura de 'Demand' no coincide con el oficial."
            )

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

    # Las ocho columnas reales de demanda (SKU -> posicion de columna).
    # Se accede por posicion (iloc) para evitar problemas con nombres
    # de columna duplicados ("Turn" aparece 3 veces, "MOTO" 3 veces, etc.).
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

    # Datos desde la fila 3 en adelante (indice 2: se saltan la decorativa
    # y la de encabezados, ya validada).
    datos = crudo.iloc[2:].reset_index(drop=True)

    registros = []
    for _, fila in datos.iterrows():
        # Leer turnos de los tres bloques por posicion.
        turns_raw = [fila.iloc[idx] for idx in POS_TURN]

        # Filas separadoras / decorativas sin turno en ningún bloque.
        if all(pd.isna(t) for t in turns_raw):
            continue

        # Validar que los turnos de los tres bloques coincidan.
        turns_int = [int(t) for t in turns_raw if not pd.isna(t)]
        if len(set(turns_int)) > 1:
            raise ValueError(
                "Los turnos de los bloques NORTE/CENTRO/SUR no coinciden: "
                f"{turns_int}. La hoja 'Demand' tiene turnos desalineados."
            )
        turn = turns_int[0]

        # Leer únicamente las ocho columnas reales de demanda.
        # Una celda vacía (NaN) se omite: no es demanda cero, es falta de dato.
        for regional, producto, pos_col in BLOQUES:
            valor = fila.iloc[pos_col]
            if pd.isna(valor):
                continue
            registros.append(
                {
                    "Turn": turn,
                    "REGIONAL": regional,
                    "PRODUCTO": producto,
                    "DEMANDA": float(valor),
                }
            )

    if not registros:
        raise ValueError("No se encontraron datos de demanda validos.")

    largo = pd.DataFrame(registros)
    return largo.sort_values("Turn").reset_index(drop=True)


def leer_costos(archivo) -> pd.DataFrame:
    """Lee la hoja 'Costos y Precios' del Excel.

    La hoja usa celdas combinadas que desplazan etiquetas; se lee por bloque
    de 4 valores numericos consecutivos (MD, MOD, Costo, Precio) y se asocian
    las 3 filas a los productos en orden estandar Moto, Cuatrimoto, Tractor
    (valida contra los valores de control de CLAUDE.md seccion 7).

    Devuelve: PRODUCTO | MD | MOD | Costo | Precio
    """
    df = pd.read_excel(archivo, sheet_name="Costos y Precios", header=None)

    filas_costos = []
    for _, fila in df.iterrows():
        bloque = None
        for i in range(len(fila) - 3):
            try:
                bloque = [float(fila[i]), float(fila[i + 1]),
                          float(fila[i + 2]), float(fila[i + 3])]
                break
            except (ValueError, TypeError):
                continue
        if bloque is not None:
            filas_costos.append(bloque)

    if len(filas_costos) < 3:
        raise ValueError(
            "No se pudieron identificar 3 filas de costos (Moto, Cuatrimoto, "
            "Tractor) en 'Costos y Precios'. Estructura inesperada."
        )

    filas_costos = filas_costos[:3]
    costos = pd.DataFrame(filas_costos, columns=["MD", "MOD", "Costo", "Precio"])
    costos.insert(0, "PRODUCTO", PRODUCTOS)
    return costos


def leer_ventas(archivo) -> pd.DataFrame:
    """Lee las hojas de ventas del Excel.

    Se esperan las hojas: VENTAS.CENTRO, VENTAS.NORTE, VENTAS.SUR.
    Cada una contiene: Turno | REGIONAL | PRODUCTO | VENTAS.

    Devuelve un DataFrame largo: Turn | REGIONAL | PRODUCTO | VENTAS
    """
    ventas = []

    for regional, sheet_nombre in [("NORTE", "VENTAS.NORTE"), ("CENTRO", "VENTAS.CENTRO"), ("SUR", "VENTAS.SUR")]:
        try:
            df = pd.read_excel(archivo, sheet_name=sheet_nombre, header=None)
        except Exception:
            continue

        if df.empty or df.shape[0] < 2:
            continue

        df_data = df.iloc[2:].reset_index(drop=True) if df.shape[0] > 2 else df.iloc[1:].reset_index(drop=True)
        encabezados_raw = df.iloc[1].fillna("").astype(str).str.strip() if df.shape[0] > 1 else pd.Series(dtype=str)

        turn_col = None
        prod_col = None
        regional_col = None
        ventas_col = None

        for idx, header in enumerate(encabezados_raw):
            h_lower = header.lower()
            if turn_col is None and any(p in h_lower for p in ["turn", "per", "periodo"]):
                turn_col = idx
            elif prod_col is None and any(p in h_lower for p in ["prod", "producto", "sku", "item", "nombre"]):
                prod_col = idx
            elif regional_col is None and any(p in h_lower for p in ["region", "reg", "loc", "ubicacion", "centro", "norte", "sur"]):
                regional_col = idx
            elif ventas_col is None and any(p in h_lower for p in ["vent", "venta", "vta", "vend"]):
                ventas_col = idx

        if turn_col is None: turn_col = 0
        if prod_col is None: prod_col = 1
        if regional_col is None: regional_col = 0
        if ventas_col is None: ventas_col = 2

        for _, fila in df_data.iterrows():
            try:
                turn = int(fila.iloc[turn_col])
            except (ValueError, TypeError):
                if pd.isna(fila.iloc[turn_col]):
                    continue
                continue

            prod = fila.iloc[prod_col]
            regional = fila.iloc[regional_col]
            try:
                venta = float(fila.iloc[ventas_col])
            except (ValueError, TypeError):
                venta = np.nan

            if pd.isna(prod) or pd.isna(venta):
                continue

            ventas.append({
                "Turn": turn,
                "REGIONAL": str(regional).strip() if pd.notna(regional) else regional,
                "PRODUCTO": str(prod).strip(),
                "VENTAS": venta,
            })

    if not ventas:
        return pd.DataFrame(columns=["Turn", "REGIONAL", "PRODUCTO", "VENTAS"])

    df_vent = pd.DataFrame(ventas)
    return df_vent.drop_duplicates().reset_index(drop=True)


def leer_inventario(archivo) -> pd.DataFrame:
    """Lee las hojas de inventario del Excel.

    Se esperan las hojas: INVENTARIO.NORTE, INVENTARIO.CENTRO, INVENTARIO.SUR.
    Cada una contiene: Turno | REGIONAL | PRODUCTO | INVENTARIO.

    Devuelve un DataFrame largo: Turn | REGIONAL | PRODUCTO | INVENTARIO
    """
    inventarios = []

    for regional, sheet_nombre in [
        ("NORTE", "INVENTARIO.NORTE"),
        ("CENTRO", "INVENTARIO.CENTRO"),
        ("SUR", "INVENTARIO.SUR"),
    ]:
        try:
            df = pd.read_excel(archivo, sheet_name=sheet_nombre, header=None)
        except Exception:
            continue

        if df.empty or df.shape[0] < 2:
            continue

        df_data = df.iloc[2:].reset_index(drop=True) if df.shape[0] > 2 else df.iloc[1:].reset_index(drop=True)
        encabezados_raw = df.iloc[1].fillna("").astype(str).str.strip() if df.shape[0] > 1 else pd.Series(dtype=str)

        turn_col = None
        prod_col = None
        regional_col = None
        inventario_col = None

        for idx, header in enumerate(encabezados_raw):
            h_lower = header.lower()
            if turn_col is None and any(p in h_lower for p in ["turn", "per", "periodo"]):
                turn_col = idx
            elif prod_col is None and any(p in h_lower for p in ["prod", "producto", "sku", "item", "nombre"]):
                prod_col = idx
            elif regional_col is None and any(p in h_lower for p in ["region", "reg", "loc", "ubicacion", "centro", "norte", "sur"]):
                regional_col = idx
            elif inventario_col is None and any(p in h_lower for p in ["invent", "inv", "stock"]):
                inventario_col = idx

        if turn_col is None: turn_col = 0
        if prod_col is None: prod_col = 1
        if regional_col is None: regional_col = 0
        if inventario_col is None: inventario_col = 2

        for _, fila in df_data.iterrows():
            try:
                turn = int(fila.iloc[turn_col])
            except (ValueError, TypeError):
                if pd.isna(fila.iloc[turn_col]):
                    continue
                continue

            prod = fila.iloc[prod_col]
            regional = fila.iloc[regional_col]
            try:
                inv = float(fila.iloc[inventario_col])
            except (ValueError, TypeError):
                inv = np.nan

            if pd.isna(prod) or pd.isna(inv):
                continue

            inventarios.append({
                "Turn": turn,
                "REGIONAL": str(regional).strip() if pd.notna(regional) else regional,
                "PRODUCTO": str(prod).strip(),
                "INVENTARIO": inv,
            })

    if not inventarios:
        return pd.DataFrame(columns=["Turn", "REGIONAL", "PRODUCTO", "INVENTARIO"])

    df_inv = pd.DataFrame(inventarios)
    return df_inv.drop_duplicates().reset_index(drop=True)


# ---------------------------------------------------------------------------
# Cálculo de indicadores financieros e inventarios (seccion 8)
# ---------------------------------------------------------------------------


def calcular_ifr_referencia(ventas: pd.DataFrame, inventario: pd.DataFrame,
                            producto: str, regional: str) -> tuple:
    """Calcula el Item Fill Rate por referencia y producto para un periodo.

    IFR = Unidades demanadas cubiertas / Unidades demanadas

    Se calcula a nivel de semana/turno para permitir el analisis
    temporal semanal.

    Devuelve: (porcentajes_por_trimestre, total_cubiertas, total_demanado)
    """
    ventas_prod = ventas[
        (ventas["PRODUCTO"] == producto) & (ventas["REGIONAL"] == regional)
    ].copy()

    inv_prod = inventario[
        (inventario["PRODUCTO"] == producto) & (inventario["REGIONAL"] == regional)
    ].copy()

    if ventas_prod.empty:
        return pd.Series(dtype=float), 0.0, 0.0

    ventas_grupo = ventas_prod.groupby("Turn")["VENTAS"].sum().sort_index()
    inv_grupo = inv_prod.groupby("Turn")["INVENTARIO"].sum().sort_index()

    fpr_trim = []

    for turn, demanda in ventas_grupo.items():
        if demanda > 0:
            inv_disp = inv_grupo.get(turn, 0.0)
            cubierto = min(inv_disp, demanda)
            ifr = cubierto / demanda
            fpr_trim.append(ifr)
        else:
            fpr_trim.append(1.0)

    return pd.Series(fpr_trim, index=ventas_grupo.index), ventas_grupo.sum(), 0.0


def calcular_ifr_global(ventas: pd.DataFrame, inventario: pd.DataFrame) -> float:
    """Calcula el IFR global para el portfolio completo.

    IFR_global = Σ unidades cubiertas / Σ unidades demanadas

    Esto evidencia la robustez/global del servicio.
    """
    total_demanado = ventas["VENTAS"].sum()
    if total_demanado == 0:
        return 1.0

    total_inventario = inventario["INVENTARIO"].sum()

    cubierto_total = min(total_inventario, total_demanado)

    ifr_global = cubierto_total / total_demanado
    return ifr_global


def calcular_inv_turnover(ventas: pd.DataFrame, inventario: pd.DataFrame,
                          regional: str) -> pd.DataFrame:
    """Calcula la rotación de inventarios por referencia.

    ROT = Costo de mercaderia vendida / Inventario promedio

    Para este análisis sin archivo de costos, usamos la demanda como
    proxy del costo de mercaderia vendida, y el inventario promedio
    por producto y regional.

    Devuelve un DataFrame con: Producto | Regional | Demanda | Inventario |
    ROT
    """
    ventas_prod = ventas[ventas["REGIONAL"] == regional].copy()
    inv_prod = inventario[inventario["REGIONAL"] == regional].copy()

    resultado = []

    for producto in PRODUCTOS:
        dem_prod = ventas_prod[ventas_prod["PRODUCTO"] == producto]["VENTAS"].sum()
        if dem_prod == 0:
            continue

        inv_prod_val = inv_prod[inv_prod["PRODUCTO"] == producto]["INVENTARIO"].sum()

        if inv_prod_val == 0:
            rot = np.nan
        else:
            rot = dem_prod / inv_prod_val

        resultado.append({
            "Producto": producto,
            "Regional": regional,
            "Demanda": dem_prod,
            "Inventario": inv_prod_val,
            "ROT": rot,
        })

    return pd.DataFrame(resultado)


def calcular_gmroi(ventas: pd.DataFrame, inventario: pd.DataFrame,
                   costos: pd.DataFrame, regional: str) -> pd.DataFrame:
    """Calcula el GMROI por referencia y producto.

    GMROI = Utilidad bruta neta / Costo promedio de inventario

    Donde:
      Utilidad bruta neta = Ventas - Costo de mercaderia
      Costo de mercaderia = Inventario x Costo unitario

    Formula alternativa:
      GMROI = (Precio - Costo) / Costo

    Devuelve un DataFrame con: Producto | Regional | Utilidad | Inventario |
    GMROI
    """
    costos_prod = costos[costos["PRODUCTO"].isin(PRODUCTOS)]

    resultado = []

    for producto in PRODUCTOS:
        cost_prod = costos_prod[costos_prod["PRODUCTO"] == producto]
        if cost_prod.empty:
            continue

        precio = float(cost_prod["Precio"].iloc[0])
        costo_unit = float(cost_prod["Costo"].iloc[0])

        ventas_prod = ventas[
            (ventas["PRODUCTO"] == producto) & (ventas["REGIONAL"] == regional)
        ]
        inv_prod = inventario[
            (inventario["PRODUCTO"] == producto) & (inventario["REGIONAL"] == regional)
        ]

        ventas_total = ventas_prod["VENTAS"].sum()
        inv_total = inv_prod["INVENTARIO"].sum()

        if ventas_total == 0 or inv_total == 0 or costo_unit == 0:
            gmroi = np.nan
            utilidad = np.nan
        else:
            utilidad = ventas_total * (precio - costo_unit)
            gmroi = utilidad / (inv_total * costo_unit)

        resultado.append({
            "Producto": producto,
            "Regional": regional,
            "Ventas": ventas_total,
            "Inventario": inv_total,
            "Precio": precio,
            "Costo_unitario": costo_unit,
            "Utilidad_bruta": utilidad,
            "GMROI": gmroi,
        })

    return pd.DataFrame(resultado)


# ---------------------------------------------------------------------------
# Graficas de indicadores (seccion 9)
# ---------------------------------------------------------------------------


def grafica_ifr_semanal_por_referencia(ifr_df: pd.DataFrame) -> go.Figure:
    """Grafica el IFR semanal por referencia y producto.

    Parametros:
      ifr_df: DataFrame con columnas [REF, PRODUCTO, TURN, IFR]

    Devuelve una figura con lineas para cada referencia/producto.
    """
    if ifr_df.empty:
        fig = go.Figure()
        fig.add_annotation(text="No hay datos suficientes para graficar IFR.",
                           x=0.5, y=0.5, showarrow=False,
                           font=dict(size=14), showgrid=False, paper_bgcolor="white")
        return fig

    fig = make_subplots(
        rows=1, cols=1,
        subplot_titles=("IFR semanal por referencia y producto",
                        "Cobertura del servicio de inventario"),
        x_title="Turno (semana)",
        y_title="IFR (0-1)"
    )

    colores = {"NORTE": "#e74c3c", "CENTRO": "#3498db", "SUR": "#2ecc71"}

    for ref in ifr_df["REF"].unique():
        datos = ifr_df[ifr_df["REF"] == ref]
        for producto in datos["PRODUCTO"].unique():
            datos_p = datos[datos["PRODUCTO"] == producto]
            if datos_p.empty:
                continue
            regional = datos_p["REGIONAL"].iloc[0]
            fig.add_trace(
                go.Scatter(
                    x=datos_p["TURN"],
                    y=datos_p["IFR"] * 100,
                    mode="lines+markers",
                    name=f"{producto} ({ref})",
                    line=dict(color=colores.get(regional, "#95a5a6"), width=2),
                    marker=dict(size=6),
                ),
                row=1, col=1
            )

    fig.update_layout(
        title="Tasa de Llenado por Producto y Referencia (IFR)",
        hovermode="x unified",
        template="plotly_white",
        height=450,
        xaxis=dict(tickmode="linear"),
    )

    fig.update_yaxes(title_text="IFR (%)", row=1, col=1)
    return fig


def grafica_ifr_global(ifr_global: float, ifr_por_referencia: dict) -> go.Figure:
    """Grafica el IFR global y por referencia.

    Parametros:
      ifr_global: valor IFR global
      ifr_por_referencia: dict con {referencia: valor}

    Devuelve un gráfico de barras comparando el IFR global con el
    IFR por referencia.
    """
    refs = list(ifr_por_referencia.keys())
    valores = [ifr_por_referencia[r] * 100 for r in refs]
    valores.append(ifr_global * 100)

    fig = go.Figure()

    fig.add_trace(
        go.Bar(
            x=refs + ["GLOBAL"],
            y=valores,
            marker_color=["#e74c3c", "#3498db", "#2ecc71", "#9b59b6"],
            text=[f"{v:.1f}%" for v in valores],
            textposition="auto",
            name="IFR",
        )
    )

    fig.update_layout(
        title="IFR Global y por Referencia",
        yaxis_title="IFR (%)",
        template="plotly_white",
        height=400,
        showlegend=False,
    )

    return fig


def preparar_datos_ifr(ventas: pd.DataFrame, inventario: pd.DataFrame) -> tuple:
    """Prepara los datos para graficar IFR por referencia y producto.

    Devuelve:
      - DataFrame largo para graficar IFR semanal [REF, PRODUCTO, REGIONAL, TURN, IFR]
      - IFR global
      - IFR por referencia dict {ref: valor}
    """
    ifr_por_producto = []

    # Alarme si no hay datos de ventas
    if ventas.empty or inventario.empty:
        return pd.DataFrame(), 1.0, {}

    for regional in ["NORTE", "CENTRO", "SUR"]:
        for producto in PRODUCTOS:
            ventas_p = ventas[
                (ventas["PRODUCTO"] == producto) & (ventas["REGIONAL"] == regional)
            ]
            if ventas_p.empty:
                continue

            inv_p = inventario[
                (inventario["PRODUCTO"] == producto) & (inventario["REGIONAL"] == regional)
            ]
            if inv_p.empty:
                continue

            # Agregar el IFR semanal
            ventas_grupo = ventas_p.groupby("Turn")["VENTAS"].sum().sort_index()
            inv_grupo = inv_p.groupby("Turn")["INVENTARIO"].sum().sort_index()

            ifr_semanal = []
            for turn, demanda in ventas_grupo.items():
                if demanda > 0:
                    inv_disp = inv_grupo.get(turn, 0.0)
                    cubierto = min(inv_disp, demanda)
                    ifr = cubierto / demanda
                    ifr_semanal.append(ifr)
                else:
                    ifr_semanal.append(1.0)

            if ifr_semanal:
                ref = f"{regional}-{producto}"
                ifr_por_producto.append({
                    "REF": ref,
                    "PRODUCTO": producto,
                    "REGIONAL": regional,
                    "TURN": ventas_grupo.index,
                    "IFR": ifr_semanal,
                })

    # Construir el DataFrame largo para graficar
    if ifr_por_producto:
        df_ifr = pd.concat([
            pd.DataFrame({
                "REF": p["REF"],
                "PRODUCTO": p["PRODUCTO"],
                "REGIONAL": p["REGIONAL"],
                "TURN": p["TURN"][i],
                "IFR": p["IFR"][i],
            })
            for p in ifr_por_producto
            for i in range(len(p["TURN"]))
        ], ignore_index=True)
    else:
        df_ifr = pd.DataFrame()

    # IFR global
    total_demanado = ventas["VENTAS"].sum()
    if total_demanado > 0:
        total_inventario = inventario["INVENTARIO"].sum()
        cubierto = min(total_inventario, total_demanado)
        ifr_global = cubierto / total_demanado
    else:
        ifr_global = 1.0

    # IFR por referencia
    ifr_por_ref = {}
    if not df_ifr.empty:
        grp = df_ifr.groupby("REF")["IFR"].mean()
        ifr_por_ref = grp.to_dict()

    return df_ifr, ifr_global, ifr_por_ref


def grafica_rotacion_inventarios(rot_df: pd.DataFrame) -> go.Figure:
    """Grafica la rotación de inventarios por referencia y producto.

    Parametros:
      rot_df: DataFrame con columnas [Producto, Regional, Demanda, Inventario, ROT]

    Devuelve un gráfico de barras agrupadas.
    """
    if rot_df.empty:
        fig = go.Figure()
        fig.add_annotation(text="No hay datos suficientes para graficar rotación.",
                           x=0.5, y=0.5, showarrow=False,
                           font=dict(size=14), showgrid=False, paper_bgcolor="white")
        return fig

    fig = go.Figure()

    # Ordenar por ROT descendente para visual
    rot_df = rot_df.sort_values("ROT", ascending=False)

    fig.add_trace(
        go.Bar(
            x=rot_df["Producto"] + " - " + rot_df["Regional"],
            y=rot_df["ROT"],
            marker_color=["#3498db", "#e74c3c", "#2ecc71", "#9b59b6", "#f39c12", "#1abc9c"][:len(rot_df)],
            text=rot_df["ROT"].apply(lambda x: f"{x:.2f}x" if not pd.isna(x) else "N/A"),
            textposition="auto",
            name="Rotación",
        )
    )

    fig.update_layout(
        title="Rotación de Inventarios por Referencia",
        yaxis_title="ROT (Ventas / Inventario)",
        template="plotly_white",
        height=400,
        xaxis_tickangle=-45,
    )

    return fig


def grafica_gmroi(gmroi_df: pd.DataFrame) -> go.Figure:
    """Grafica el GMROI por referencia y producto.

    Parametros:
      gmroi_df: DataFrame con columnas [Producto, Regional, Utilidad_bruta, Inventario, GMROI]

    Devuelve un gráfico de barras con GMROI.
    """
    if gmroi_df.empty:
        fig = go.Figure()
        fig.add_annotation(text="No hay datos suficientes para graficar GMROI.",
                           x=0.5, y=0.5, showarrow=False,
                           font=dict(size=14), showgrid=False, paper_bgcolor="white")
        return fig

    # Ordenar por GMROI descendente
    gmroi_df = gmroi_df.sort_values("GMROI", ascending=False)

    fig = go.Figure()

    fig.add_trace(
        go.Bar(
            x=gmroi_df["Producto"] + " - " + gmroi_df["Regional"],
            y=gmroi_df["GMROI"],
            marker_color=["#3498db", "#e74c3c", "#2ecc71", "#9b59b6", "#f39c12", "#1abc9c"][:len(gmroi_df)],
            text=gmroi_df["GMROI"].apply(lambda x: f"{x:.2f}x" if not pd.isna(x) else "N/A"),
            textposition="auto",
            name="GMROI",
        )
    )

    fig.update_layout(
        title="GMROI por Referencia y Producto",
        yaxis_title="GMROI",
        template="plotly_white",
        height=400,
        xaxis_tickangle=-45,
    )

    return fig


def calcular_rotacion_gmroi_comparacion(ventas: pd.DataFrame, inventario: pd.DataFrame,
                                        costos: pd.DataFrame) -> pd.DataFrame:
    """Calcula rotación y GMROI combinados para una comparación cruzada.

    Devuelve un DataFrame con: Producto | Regional | Rotación | GMROI |
    Utilidad_bruta | Inventario
    """
    # Usar los cálculos anteriores
    rot_df = calcular_inv_turnover(ventas, inventario, regional="ALL" if False else "CENTRO")
    # TODO: This is a placeholder, the full comparison is done in the app

    # We'll compute per-regional and combine
    results = []
    regionals = ["NORTE", "CENTRO", "SUR"]

    for regional in regionals:
        rot_p = calcular_inv_turnover(ventas, inventario, regional)
        gmroi_p = calcular_gmroi(ventas, inventario, costos, regional)

        for _, row_rot in rot_p.iterrows():
            producto = row_rot["Producto"]
            inv_val = row_rot["Inventario"]

            # Get GMROI for same product/regional
            gmroi_row = gmroi_p[gmroi_p["Producto"] == producto]
            gmroi_val = gmroi_row["GMROI"].iloc[0] if not gmroi_row.empty else np.nan
            utilidad_val = gmroi_row["Utilidad_bruta"].iloc[0] if not gmroi_row.empty else np.nan

            results.append({
                "Producto": producto,
                "Regional": regional,
                "Demanda": row_rot["Demanda"],
                "Inventario": inv_val,
                "ROT": row_rot["ROT"],
                "Utilidad_bruta": utilidad_val,
                "GMROI": gmroi_val,
            })

    return pd.DataFrame(results)
    """Lee las hojas de inventario del Excel.

    Se esperan las hojas: INV.CENTRO, INV.NORTE, INV.SUR.
    Cada una contiene referencia | producto | inventario.

    Devuelve un DataFrame combinado:
    REF | PRODUCTO | REGIONAL | INVENTARIO
    """
    inventario = []

    for regional, sheet_nombre in [("NORTE", "INV.NORTE"), ("CENTRO", "INV.CENTRO"), ("SUR", "INV.SUR")]:
        try:
            df = pd.read_excel(archivo, sheet_name=sheet_nombre, header=None)
        except Exception:
            continue

        if df.empty or df.shape[0] < 2:
            continue

        df_data = df.iloc[2:].reset_index(drop=True)
        encabezados = df.iloc[1].fillna("").astype(str).str.strip()

        col_inv = None
        ref_col = None
        prod_col = None
        regional_col = None

        for idx, header in enumerate(encabezados):
            h_lower = header.lower()
            if ref_col is None and any(p in h_lower for p in ["ref", "referencia", "articulo", "item"]):
                ref_col = idx
            elif prod_col is None and any(p in h_lower for p in ["prod", "producto", "sku", "nombre"]):
                prod_col = idx
            elif regional_col is None and any(p in h_lower for p in ["region", "reg", "loc", "ubicacion", "centro", "norte", "sur"]):
                regional_col = idx
            elif col_inv is None and any(p in h_lower for p in ["inv", "stock", "existencia", "inventor"]):
                col_inv = idx

        if ref_col is None: ref_col = 0
        if prod_col is None: prod_col = 1
        if regional_col is None: regional_col = 0

        for _, fila in df_data.iterrows():
            ref = fila.iloc[ref_col] if ref_col < len(fila) else None
            prod = fila.iloc[prod_col] if prod_col < len(fila) else None
            regional = fila.iloc[regional_col] if regional_col < len(fila) else regional

            if pd.isna(ref) or pd.isna(prod):
                continue

            try:
                inv_val = float(fila.iloc[col_inv]) if col_inv is not None and col_inv < len(fila) else np.nan
            except (ValueError, TypeError):
                inv_val = np.nan

            inventario.append({
                "REF": str(ref).strip(),
                "PRODUCTO": str(prod).strip(),
                "REGIONAL": str(regional).strip() if pd.notna(regional) else regional,
                "INVENTARIO": inv_val,
            })

    if not inventario:
        return pd.DataFrame(columns=["REF", "PRODUCTO", "REGIONAL", "INVENTARIO"])

    df_inv = pd.DataFrame(inventario)
    return df_inv.drop_duplicates().reset_index(drop=True)


def validar_datos(demanda: pd.DataFrame, costos: pd.DataFrame) -> dict:
    """Valida la estructura de los datos antes de calcular (seccion 6).

    No corrige datos silenciosamente. Devuelve el estado de cada control.
    """
    controles = []
    ok = True

    # 1) Exactamente 8 SKU en demanda.
    skus = set(zip(demanda["REGIONAL"], demanda["PRODUCTO"]))
    tiene_8 = skus == set(SKU_UNIVERSO)
    controles.append(("Exactamente 8 SKU (sin Tractor-Sur)", tiene_8))
    ok = ok and tiene_8

    # 2) No existe Tractor-Sur.
    sin_tractor_sur = ("SUR", "TRACTOR") not in skus
    controles.append(("No existe Tractor-Sur", sin_tractor_sur))
    ok = ok and sin_tractor_sur

    # 3) Tres productos en costos.
    tres = set(costos["PRODUCTO"]) == set(PRODUCTOS)
    controles.append(("Tres productos en costos (Moto, Cuatrimoto, Tractor)", tres))
    ok = ok and tres

    # 4) Costo unitario = MD + MOD (seccion 7).
    costo_cuadra = np.allclose(
        costos["Costo"].astype(float),
        costos["MD"].astype(float) + costos["MOD"].astype(float),
    )
    controles.append(("Costo unitario = MD + MOD", costo_cuadra))
    ok = ok and costo_cuadra

    # 5) Campos numericos validos (sin NaN en demanda o costos).
    demanda_valida = bool(demanda["DEMANDA"].notna().all())
    controles.append(("Campos numericos de demanda validos", demanda_valida))
    ok = ok and demanda_valida

    # 6) Sin duplicados de Turn por SKU.
    dup = demanda.duplicated(subset=["Turn", "REGIONAL", "PRODUCTO"]).any()
    controles.append(("Sin duplicados de Turn por SKU", not bool(dup)))
    ok = ok and not bool(dup)

    # 7) Orden temporal correcto (Turn consecutivo).
    turns = sorted(demanda["Turn"].unique())
    orden_ok = turns == list(range(min(turns), max(turns) + 1))
    controles.append(("Turn consecutivo (orden temporal correcto)", orden_ok))
    ok = ok and orden_ok

    return {"ok": ok, "controles": controles}


def construir_agregados(demanda: pd.DataFrame) -> pd.DataFrame:
    """Construye los agregados oficiales de analisis/pronostico (seccion 5.2).

    CEDI      = Centro + Sur
    MOTOTRAK  = Norte + Centro + Sur

    Se suman por producto. NO son SKU del ABC-XYZ.
    """
    base = demanda.copy()

    cedi = base[base["REGIONAL"].isin(["CENTRO", "SUR"])].copy()
    cedi = cedi.groupby(["Turn", "PRODUCTO"], as_index=False)["DEMANDA"].sum()
    cedi["REGIONAL"] = "CEDI"

    moto = base.groupby(["Turn", "PRODUCTO"], as_index=False)["DEMANDA"].sum()
    moto["REGIONAL"] = "MOTOTRAK"

    agregados = pd.concat([cedi, moto], ignore_index=True)
    return agregados[["Turn", "REGIONAL", "PRODUCTO", "DEMANDA"]]


# ---------------------------------------------------------------------------
# Motor economico y ABC (secciones 8 y 9)
# ---------------------------------------------------------------------------

def calcular_abc(demanda: pd.DataFrame, costos: pd.DataFrame,
                 ventana: int = VENTANA_ANUAL_DEFAULT) -> tuple:
    """Calcula el motor economico y la clasificacion ABC (seccion 9).

    Ventana movil anual (seccion 8): usa los ultimos 'ventana' turnos.

    Para cada SKU:
      1. totalizar demanda del ultimo ano;
      2. asociar costo y precio;
      3. calcular utilidad unitaria = Precio - Costo unitario;
      4. calcular utilidad total = Demanda anual x Utilidad unitaria;
      5. ordenar de mayor a menor utilidad total;
      6. calcular participacion;
      7. calcular acumulado;
      8. asignar ABC (A: <=80%; B: >80% y <=95%; C: >95%).

    No fuerza Pareto 80/20. Devuelve (tabla_abc, info_ventana).
    """
    # Ventana movil anual: ultimos 'ventana' turnos disponibles.
    turn_max = int(demanda["Turn"].max())
    turn_min_ventana = max(int(demanda["Turn"].min()), turn_max - ventana + 1)
    dem_ventana = demanda[demanda["Turn"] >= turn_min_ventana].copy()

    info_ventana = {
        "turn_inicial": turn_min_ventana,
        "turn_final": turn_max,
        "observaciones": int(dem_ventana["Turn"].nunique()),
    }

    # 1) Demanda anual por SKU.
    dem_anual = (
        dem_ventana.groupby(["REGIONAL", "PRODUCTO"], as_index=False)["DEMANDA"]
        .sum()
        .rename(columns={"DEMANDA": "Demanda anual"})
    )

    # 2-4) Asociar costo/precio y calcular utilidades.
    abc = dem_anual.merge(costos, on="PRODUCTO", how="left")
    abc["Utilidad unitaria"] = abc["Precio"].astype(float) - abc["Costo"].astype(float)
    abc["Utilidad total"] = abc["Demanda anual"] * abc["Utilidad unitaria"]

    # 5) Ordenar de mayor a menor utilidad total.
    abc = abc.sort_values("Utilidad total", ascending=False).reset_index(drop=True)

    # 6-7) Participacion y acumulado sobre la utilidad total del portafolio.
    total_portafolio = abc["Utilidad total"].sum()
    abc["Participacion"] = abc["Utilidad total"] / total_portafolio
    abc["Acumulado"] = abc["Participacion"].cumsum()

    # 8) Asignar ABC (sin mover manualmente el SKU que cruza un umbral).
    def _clase(acum):
        if acum <= 0.80:
            return "A"
        elif acum <= 0.95:
            return "B"
        return "C"

    abc["ABC"] = abc["Acumulado"].apply(_clase)

    # Ranking y SKU legible.
    abc.insert(0, "Ranking", abc.index + 1)
    abc["SKU"] = abc["PRODUCTO"] + "-" + abc["REGIONAL"]

    # Reordenamos columnas para la tabla minima (seccion 9).
    columnas = [
        "Ranking", "SKU", "REGIONAL", "PRODUCTO", "Demanda anual",
        "MD", "MOD", "Costo", "Precio", "Utilidad unitaria",
        "Utilidad total", "Participacion", "Acumulado", "ABC",
    ]
    abc = abc[columnas]
    return abc, info_ventana


# ---------------------------------------------------------------------------
# Motor de pronosticos (secciones 10-14)
# ---------------------------------------------------------------------------

def preparar_series(demanda: pd.DataFrame) -> pd.DataFrame:
    """Convierte la demanda al formato StatsForecast (seccion 10).

      - unique_id = "REGIONAL|PRODUCTO"
      - ds = Turn convertido a fechas ficticias semanales (freq="W-MON")
      - y = demanda
    """
    df = demanda.copy()
    df["unique_id"] = df["REGIONAL"] + "|" + df["PRODUCTO"]
    # Fecha ficticia semanal: turno 1 -> primera semana (lunes de referencia).
    df["ds"] = pd.to_datetime("2020-01-06") + pd.to_timedelta(
        df["Turn"] - 1, unit="W"
    )
    df = df.rename(columns={"DEMANDA": "y"})
    return df[["unique_id", "ds", "y"]].sort_values(["unique_id", "ds"])


def obtener_modelos():
    """Devuelve la lista de modelos oficiales (seccion 11).

    No agregar/eliminar modelos sin autorizacion.
    """
    return [
        HoltWinters(season_length=SEASON_LENGTH, alias="hw"),
        WindowAverage(window_size=3, alias="wa_3"),
        WindowAverage(window_size=6, alias="wa_6"),
        WindowAverage(window_size=12, alias="wa_12"),
        SimpleExponentialSmoothingOptimized(alias="ses"),
        MSTL(season_length=SEASON_LENGTH, alias="mstl"),
    ]


def calcular_metricas(y_real: np.ndarray, y_pred: np.ndarray) -> dict:
    """Calcula las metricas oficiales (seccion 12).

    MAE%   = sum(|y - pred|) / sum(y)
    Sesgo% = sum(y - pred) / sum(y)      (Error = Real - Pronostico)
    Score% = MAE% + |Sesgo%|
    RMSE   = sqrt(mean((y - pred)^2))

    No sustituir MAE% por MAPE.
    """
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
    return {
        "MAE%": mae_pct,
        "Sesgo%": sesgo_pct,
        "Score%": score_pct,
        "RMSE": rmse,
    }


def ejecutar_backtesting(df_ts: pd.DataFrame, n_windows: int, h: int,
                         progreso=None) -> tuple:
    """Ejecuta el motor de pronosticos (secciones 10-14).

    Para cada serie (SKU o agregado):
      1. evaluar los modelos oficiales aplicables;
      2. calcular metricas por validacion cruzada;
      3. seleccionar el de menor Score% (ganador);
      4. reentrenar el ganador;
      5. generar h periodos futuros.

    Validacion cruzada oficial:
        cv = cross_validation(df=df_ts, h=h, n_windows=n_windows, step_size=h)

    Una falla individual de un modelo NO detiene el motor: se registra la
    causa y se continua. Devuelve (comparacion, ganadores, pronosticos, errores).
    """
    modelos = obtener_modelos()
    alias_modelos = [m.alias for m in modelos]

    series_ids = list(df_ts["unique_id"].unique())
    total = len(series_ids)

    comparacion_registros = []
    ganadores = {}
    pronosticos = {}
    errores = []

    for idx, uid in enumerate(series_ids):
        if progreso is not None:
            progreso(idx / total, f"Procesando serie {idx + 1}/{total}: {uid}")

        sub = df_ts[df_ts["unique_id"] == uid][["unique_id", "ds", "y"]].copy()

        # Metricas por validacion cruzada para cada modelo.
        metricas_serie = {}
        try:
            # cross_validation es un METODO del objeto StatsForecast
            # (CLAUDE.md seccion 10): cv = sf.cross_validation(df=df_ts, h=h,
            # n_windows=n_windows, step_size=h). step_size=h -> ventanas no
            # solapadas.
            sf = StatsForecast(models=modelos, freq="W-MON", n_jobs=-1)
            cv = sf.cross_validation(
                df=sub, h=h, step_size=h, n_windows=n_windows,
            )
            for alias in alias_modelos:
                if alias not in cv.columns:
                    continue
                mask = cv[alias].notna()
                if not mask.any():
                    continue
                m = calcular_metricas(
                    cv["y"].values[mask.values],
                    cv[alias].values[mask.values],
                )
                metricas_serie[alias] = m
                comparacion_registros.append({
                    "unique_id": uid, "MODELO": alias,
                    "MAE%": m["MAE%"], "Sesgo%": m["Sesgo%"],
                    "Score%": m["Score%"], "RMSE": m["RMSE"],
                })
        except Exception as e:  # una falla no detiene el motor
            errores.append(f"{uid}: error en backtesting -> {e}")

        # --- Paso 3: seleccionar ganador por menor Score%.
        if metricas_serie:
            # Desempate exacto: RMSE como criterio secundario.
            ganador = min(
                metricas_serie.items(),
                key=lambda kv: (kv[1]["Score%"], kv[1]["RMSE"]),
            )
            alias_ganador, metrica_ganador = ganador
            ganadores[uid] = {
                "MODELO": alias_ganador,
                **metrica_ganador,
            }
        else:
            alias_ganador = None
            errores.append(f"{uid}: ningun modelo produjo metricas validas")

        # --- Paso 4-5: reentrenar ganador y pronosticar h periodos.
        if alias_ganador is not None:
            try:
                modelo_ganador = next(
                    m for m in modelos if m.alias == alias_ganador
                )
                sf_g = StatsForecast(
                    models=[modelo_ganador], freq="W-MON", n_jobs=-1
                )
                fc = sf_g.forecast(df=sub, h=h)
                fc = fc.reset_index()
                # Columnas: unique_id, ds, <alias>
                fc = fc.rename(columns={alias_ganador: "yhat"})
                pronosticos[uid] = fc[["ds", "yhat"]]
            except Exception as e:
                errores.append(f"{uid}: error al pronosticar -> {e}")

    comparacion = pd.DataFrame(comparacion_registros)
    return comparacion, ganadores, pronosticos, errores


# ---------------------------------------------------------------------------
# Clasificacion XYZ y combinacion ABC-XYZ (secciones 15, 16, 17)
# ---------------------------------------------------------------------------

def clasificar_xyz(ganadores: dict, limite_xy: float, limite_yz: float) -> dict:
    """Clasifica XYZ usando el Score% del modelo ganador de cada SKU (seccion 15).

    Regla:
      - Score% <= X/Y           -> X (mayor pronosticabilidad)
      - X/Y < Score% <= Y/Z     -> Y (intermedio)
      - Score% > Y/Z            -> Z (menor pronosticabilidad)

    No usa CV. Valida limite X/Y < limite Y/Z. Devuelve {uid: 'X'|'Y'|'Z'}.
    """
    if limite_xy >= limite_yz:
        raise ValueError(
            "Umbrales XYZ inconsistentes: se requiere limite X/Y < limite Y/Z."
        )

    xyz = {}
    for uid, info in ganadores.items():
        score = info.get("Score%", np.nan)
        if pd.isna(score):
            # Un error tecnico NO se clasifica como Z (seccion 29).
            xyz[uid] = None
            continue
        if score <= limite_xy:
            xyz[uid] = "X"
        elif score <= limite_yz:
            xyz[uid] = "Y"
        else:
            xyz[uid] = "Z"
    return xyz


def combinar_abc_xyz(abc: pd.DataFrame, ganadores: dict, xyz: dict) -> pd.DataFrame:
    """Combina ABC + XYZ sobre exactamente los 8 SKU (seccion 16).

    CEDI/MOTOTRAK no aparecen como SKU. Devuelve la tabla final.
    """
    # abc contiene SOLO los 8 SKU (se construyo sobre el universo).
    filas = []
    for _, r in abc.iterrows():
        uid = f"{r['REGIONAL']}|{r['PRODUCTO']}"
        g = ganadores.get(uid, {})
        clase_xyz = xyz.get(uid)
        abc_xyz = f"{r['ABC']}{clase_xyz}" if clase_xyz else f"{r['ABC']}?"
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
            "ABC-XYZ": abc_xyz,
        })

    return pd.DataFrame(filas)


def consolidado_producto(demanda: pd.DataFrame, costos: pd.DataFrame,
                         ventana: int) -> pd.DataFrame:
    """Consolida demanda y utilidad por producto (seccion 17).

    Evalua Moto, Cuatrimoto y Tractor de forma agregada. No altera ABC-XYZ.
    """
    turn_max = int(demanda["Turn"].max())
    turn_min = max(int(demanda["Turn"].min()), turn_max - ventana + 1)
    dem = demanda[demanda["Turn"] >= turn_min]

    dem_prod = (
        dem.groupby("PRODUCTO", as_index=False)["DEMANDA"].sum()
        .rename(columns={"DEMANDA": "Demanda anual"})
    )
    cons = dem_prod.merge(costos, on="PRODUCTO", how="left")
    cons["Utilidad unitaria"] = cons["Precio"].astype(float) - cons["Costo"].astype(float)
    cons["Utilidad total"] = cons["Demanda anual"] * cons["Utilidad unitaria"]
    total = cons["Utilidad total"].sum()
    cons["Participacion %"] = (cons["Utilidad total"] / total) * 100
    cons = cons.sort_values("Utilidad total", ascending=False).reset_index(drop=True)
    return cons


# ---------------------------------------------------------------------------
# Modelo de gestion de inventarios: revision periodica (R, S)
# ---------------------------------------------------------------------------

def disenar_revision_periodica(ganadores: dict, pronosticos: dict,
                               R: int, L: int, Z: float) -> pd.DataFrame:
    """Disena un modelo de inventarios de revision periodica (R, S).

    Cada R turnos se revisa el inventario y se reordena hasta el nivel
    objetivo S. Usa los PRONOSTICOS del modelo ganador y el RMSE del motor
    (NO el promedio ni la desviacion estandar del historico):

      Periodo de proteccion:   P = R + L
      Demanda esperada en P:   suma de los primeros P pronosticos del ganador
      Incertidumbre en P:      sigma_P = RMSE_ganador * sqrt(P)
      Stock de seguridad:      SS = Z * sigma_P
      Nivel objetivo:          S  = Demanda_P + SS

    R = periodo de revision; L = lead time (tiempo de entrega);
    Z = factor de nivel de servicio (p. ej. 1.645 ~ 95%).

    Devuelve una fila por referencia (los 8 SKU + CEDI). Si P supera el
    horizonte pronosticado, la demanda en P no se puede cubrir con los
    pronosticos disponibles y se marca (aumente h en parametros).
    """
    import math

    # Regionales solicitadas: Norte, Centro, Sur y CEDI (NO MOTOTRAK).
    regionales_interes = {"NORTE", "CENTRO", "SUR", "CEDI"}
    P = int(R) + int(L)
    filas = []
    for uid, info in ganadores.items():
        regional = uid.split("|")[0]
        if regional not in regionales_interes:
            continue  # excluye MOTOTRAK (no solicitado para el modelo R,S)
        rmse = info.get("RMSE", np.nan)
        modelo = info.get("MODELO", "N/A")
        fc = pronosticos.get(uid)
        yhat = fc["yhat"].values if fc is not None else np.array([])

        # Demanda esperada durante P: suma de los primeros P pronosticos.
        if P > 0 and len(yhat) >= P:
            demanda_p = float(np.sum(yhat[:P]))
            cubre = True
        else:
            demanda_p = np.nan
            cubre = False

        # Incertidumbre propagada desde el RMSE del motor.
        if not pd.isna(rmse) and P > 0:
            sigma_p = float(rmse) * math.sqrt(P)
        else:
            sigma_p = np.nan

        ss = Z * sigma_p if not pd.isna(sigma_p) else np.nan
        nivel_s = (demanda_p + ss) if (cubre and not pd.isna(ss)) else np.nan

        regional, producto = uid.split("|")
        filas.append({
            "Regional": regional,
            "Producto": producto,
            "Modelo ganador": modelo,
            "R (revision)": int(R),
            "L (entrega)": int(L),
            "P = R+L": P,
            "Demanda en P (suma pronosticos)": demanda_p,
            "RMSE (motor)": rmse,
            "sigma_P = RMSE*sqrt(P)": sigma_p,
            "Stock de seguridad": ss,
            "Nivel objetivo S": nivel_s,
            "Cubre horizonte": cubre,
        })

    df = pd.DataFrame(filas)
    # Orden amigable: Norte, Centro, Sur, CEDI; y por producto.
    orden = {"NORTE": 0, "CENTRO": 1, "SUR": 2, "CEDI": 3}
    if not df.empty:
        df["_o"] = df["Regional"].map(orden).fillna(9)
        df = df.sort_values(["_o", "Producto"]).drop(columns="_o")
        df = df.reset_index(drop=True)
    return df



# ---------------------------------------------------------------------------
# Exportacion a Excel (seccion 21)
# ---------------------------------------------------------------------------

def generar_excel(abc: pd.DataFrame, costos: pd.DataFrame, abc_xyz: pd.DataFrame,
                  comparacion: pd.DataFrame, pronosticos: dict,
                  info_ventana: dict, n_windows: int, h: int,
                  limite_xy: float, limite_yz: float,
                  conclusiones: list, ruta_salida: str,
                  rs_tabla: pd.DataFrame = None) -> None:
    """Genera el Excel de resultados sin sobrescribir la fuente (seccion 21).

    Hojas minimas: Resumen, Costos, Clasif ABC_XYZ, Pronosticos,
    Comparacion_Modelos, Conclusiones. Si se provee, ademas Revision_Periodica.
    """
    # --- Hoja Resumen: parametros y conteos.
    utilidad_total = float(abc["Utilidad total"].sum())
    conteos_abc = abc["ABC"].value_counts().to_dict()
    conteos_xyz = (abc_xyz["XYZ"].value_counts().to_dict()
                   if not abc_xyz.empty else {})
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

    # --- Hoja Pronosticos: valores futuros por serie (turnos futuros).
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

    # --- Hoja Conclusiones.
    concl_df = pd.DataFrame(conclusiones, columns=["Hallazgo / Observacion"])

    with pd.ExcelWriter(ruta_salida, engine="openpyxl") as writer:
        resumen.to_excel(writer, sheet_name="Resumen", index=False)
        costos.to_excel(writer, sheet_name="Costos", index=False)
        abc_xyz.to_excel(writer, sheet_name="Clasif ABC_XYZ", index=False)
        pronosticos_df.to_excel(writer, sheet_name="Pronosticos", index=False)
        comparacion.to_excel(writer, sheet_name="Comparacion_Modelos",
                             index=False)
        concl_df.to_excel(writer, sheet_name="Conclusiones", index=False)
        if rs_tabla is not None and not rs_tabla.empty:
            rs_tabla.to_excel(writer, sheet_name="Revision_Periodica_RS",
                              index=False)


    # --- Agregar indicadores adicionales al Excel
    # (IFR, Rotacion e GMROI) si hay datos de ventas e inventario.
    try:
        ventas = st.session_state.get("_ventas", None)
        inventario = st.session_state.get("_inventario", None)

        if ventas is not None and not ventas.empty and inventario is not None and not inventario.empty:
            import openpyxl
            import shutil
            import os

            # Copiar el archivo original
            shutil.copy(ruta_salida, ruta_salida + ".tmp")

            # Abrir el archivo y agregar hojas adicionales
            with pd.ExcelWriter(ruta_salida + ".tmp", engine="openpyxl",
                                mode="a", if_sheet_exists="replace") as writer:
                # IFR semanal
                ifr_df, ifr_global, ifr_por_ref = preparar_datos_ifr(ventas, inventario)
                if not ifr_df.empty:
                    ifr_long = ifr_df.melt(id_vars=["REF", "PRODUCTO", "REGIONAL", "TURN"],
                                             value_vars=["IFR"],
                                             var_name="Metrica",
                                             value_name="Valor")
                    ifr_long.to_excel(writer, sheet_name="IFR_Semanal", index=False)

                    ifr_resumen = pd.DataFrame({
                        "Metrica": ["IFR Global",
                                    "IFR NORTE-MOTO", "IFR CENTRO-MOTO", "IFR SUR-MOTO"],
                        "Valor": [
                            ifr_global * 100,
                            ifr_por_ref.get("NORTE-MOTO", np.nan) * 100,
                            ifr_por_ref.get("CENTRO-MOTO", np.nan) * 100,
                            ifr_por_ref.get("SUR-MOTO", np.nan) * 100,
                        ]
                    })
                    ifr_resumen.to_excel(writer, sheet_name="IFR_Resumen", index=False)

                # Rotacion de inventarios
                rot_df = calcular_rotacion_gmroi_comparacion(ventas, inventario, costos)
                if not rot_df.empty:
                    rot_df.to_excel(writer, sheet_name="Rotacion_Inventarios", index=False)

                # GMROI
                gmroi_df = calcular_gmroi(ventas, inventario, costos, None)
                if not gmroi_df.empty:
                    gmroi_df.to_excel(writer, sheet_name="GMROI", index=False)

            # Reemplazar el archivo original
            os.remove(ruta_salida)
            os.rename(ruta_salida + ".tmp", ruta_salida)

    except Exception as e:
        # Si hay error, restaurar el archivo original
        if os.path.exists(ruta_salida + ".tmp"):
            os.remove(ruta_salida + ".tmp")
        st.warning(f"No se pudieron agregar las hojas adicionales al Excel: {e}")

# ---------------------------------------------------------------------------
# Visualizaciones (secciones 19 y 20)

# ---------------------------------------------------------------------------
# Visualizaciones (secciones 19 y 20)
# ---------------------------------------------------------------------------

def grafica_pronosticos_oficial(df_ts: pd.DataFrame, pronosticos: dict,
                                h: int, turn_max: int) -> go.Figure:
    """Visualizacion oficial de pronosticos (seccion 19).

    Subplots 2x3: NORTE, CENTRO, SUR, CEDI (C+S), MOTOTRAK (N+C+S), sexto vacio.
    Linea solida = real; punteada = pronostico. Ultimos 52+h turnos.
    Colores: MOTO=salmon, CUATRIMOTO=navy, TRACTOR=darkcyan. Leyenda una vez.

    df_ts debe incluir los 8 SKU y los agregados CEDI/MOTOTRAK como series
    con unique_id "REGIONAL|PRODUCTO".
    """
    titulos = ["NORTE", "CENTRO", "SUR", "CEDI (C+S)", "MOTOTRAK (N+C+S)", ""]
    fig = make_subplots(rows=2, cols=3, subplot_titles=titulos)

    # Panel -> (row, col) y lista de unique_id a graficar.
    # Cada panel agrupa por PRODUCTO (MOTO, CUATRIMOTO, TRACTOR).
    paneles = {
        (1, 1): ["NORTE|MOTO", "NORTE|CUATRIMOTO", "NORTE|TRACTOR"],
        (1, 2): ["CENTRO|MOTO", "CENTRO|CUATRIMOTO", "CENTRO|TRACTOR"],
        (1, 3): ["SUR|MOTO", "SUR|CUATRIMOTO"],  # Sur no tiene Tractor
        (2, 1): ["CEDI|MOTO", "CEDI|CUATRIMOTO", "CEDI|TRACTOR"],
        (2, 2): ["MOTOTRAK|MOTO", "MOTOTRAK|CUATRIMOTO", "MOTOTRAK|TRACTOR"],
    }

    ventana_inicio = turn_max - (VENTANA_ANUAL_DEFAULT + h) + 1
    fecha_corte = pd.to_datetime("2020-01-06") + pd.to_timedelta(
        ventana_inicio - 1, unit="W"
    )

    leyenda_agregada = set()

    for (row, col), uids in paneles.items():
        for uid in uids:
            producto = uid.split("|")[1]
            color = COLOR_PRODUCTO[producto]
            show = producto not in leyenda_agregada
            leyenda_agregada.add(producto)

            # Serie real (ultimos 52+h turnos).
            d = df_ts[df_ts["unique_id"] == uid]
            d = d[d["ds"] >= fecha_corte]
            if not d.empty:
                fig.add_trace(
                    go.Scatter(x=d["ds"], y=d["y"], mode="lines",
                               line=dict(color=color, width=2),
                               name=producto, legendgroup=producto,
                               showlegend=show),
                    row=row, col=col,
                )
            # Pronostico (linea punteada).
            if uid in pronosticos:
                fc = pronosticos[uid]
                fig.add_trace(
                    go.Scatter(x=fc["ds"], y=fc["yhat"], mode="lines",
                               line=dict(color=color, width=2, dash="dash"),
                               name=producto, legendgroup=producto,
                               showlegend=False),
                    row=row, col=col,
                )

    fig.update_layout(
        template="ggplot2", height=700,
        title="Pronosticos por region y agregados (real vs modelo ganador)",
    )
    fig.update_xaxes(title_text="Turn")
    fig.update_yaxes(title_text="Demanda")
    return fig



def grafica_pareto(abc: pd.DataFrame) -> go.Figure:
    """Pareto ABC complementario (seccion 20)."""
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(
        go.Bar(x=abc["SKU"], y=abc["Participacion"] * 100, name="Participacion %",
               marker_color="steelblue"),
        secondary_y=False,
    )
    fig.add_trace(
        go.Scatter(x=abc["SKU"], y=abc["Acumulado"] * 100, name="Acumulado %",
                   mode="lines+markers", line=dict(color="firebrick")),
        secondary_y=True,
    )
    fig.update_layout(template="ggplot2", title="Pareto ABC (por Utilidad total)",
                      height=450)
    fig.update_yaxes(title_text="Participacion %", secondary_y=False)
    fig.update_yaxes(title_text="Acumulado %", secondary_y=True)
    return fig


def grafica_score(abc_xyz: pd.DataFrame) -> go.Figure:
    """Score% de los 8 SKU (seccion 20)."""
    fig = go.Figure()
    colores_xyz = {"X": "seagreen", "Y": "goldenrod", "Z": "indianred"}
    fig.add_trace(go.Bar(
        x=abc_xyz["SKU"], y=abc_xyz["Score%"] * 100,
        marker_color=[colores_xyz.get(z, "gray") for z in abc_xyz["XYZ"]],
        name="Score%",
    ))
    fig.update_layout(template="ggplot2",
                      title="Score% del modelo ganador por SKU (color = XYZ)",
                      height=420, yaxis_title="Score%")
    return fig


def grafica_matriz_abcxyz(abc_xyz: pd.DataFrame) -> go.Figure:
    """Matriz ABC-XYZ (seccion 20)."""
    clases = ["A", "B", "C"]
    ejes = ["X", "Y", "Z"]
    z = [[0] * 3 for _ in range(3)]
    texto = [["" for _ in range(3)] for _ in range(3)]
    for _, r in abc_xyz.iterrows():
        if r["ABC"] in clases and r["XYZ"] in ejes:
            i = clases.index(r["ABC"])
            j = ejes.index(r["XYZ"])
            z[i][j] += 1
            texto[i][j] += r["SKU"] + "\n"
    fig = go.Figure(data=go.Heatmap(
        z=z, x=ejes, y=clases,
        text=texto, texttemplate="%{text}", hoverinfo="text",
        colorscale="Blues", showscale=False,
    ))
    fig.update_layout(template="ggplot2", title="Matriz ABC-XYZ (conteo de SKU)",
                      height=400, yaxis=dict(autorange="reversed"))
    return fig


# ---------------------------------------------------------------------------
# Interfaz Streamlit (seccion 18)
# ---------------------------------------------------------------------------

@st.cache_data(show_spinner="Ejecutando motor de pronosticos (puede tardar)...")
def _motor_cacheado(demanda: pd.DataFrame, n_windows: int, h: int):
    """Ejecuta el motor y lo cachea por (demanda, n_windows, h).

    Asi, mover los limites XYZ u otras opciones de visualizacion NO vuelve a
    ejecutar el costoso backtesting: solo se recalcula si cambian los datos o
    los parametros del motor. Devuelve ademas el df_ts para los graficos.
    """
    agregados = construir_agregados(demanda)
    df_ts = preparar_series(pd.concat([demanda, agregados], ignore_index=True))
    comparacion, ganadores, pronosticos, errores = ejecutar_backtesting(
        df_ts, n_windows=n_windows, h=h
    )
    return comparacion, ganadores, pronosticos, errores, df_ts


def main():
    st.set_page_config(page_title="Mototrack", layout="wide")
    st.title("Mototrack")
    st.markdown(
        "Aplicacion local para la Actividad 2 de Gestion de Inventarios. "
        "Todos los calculos son trazables y verificables segun la metodologia."
    )

    st.sidebar.header("1. Carga de datos")
    archivo = st.sidebar.file_uploader(
        "Cargar Excel de Mototrack (.xlsx)", type=["xlsx"]
    )

    if archivo is None:
        st.info("Carga el archivo Excel de Mototrack para comenzar. "
                "Debe contener las hojas 'Demand' y 'Costos y Precios'.")
        return

    st.session_state["nombre_archivo"] = archivo.name

    # --- Paso 1: lectura y validacion.
    try:
        demanda = leer_demanda(archivo)
        costos = leer_costos(archivo)
        ventas = leer_ventas(archivo)
        inventario = leer_inventario(archivo)
    except Exception as e:
        st.error(f"Error al leer el archivo: {e}")
        return

    st.session_state["_ventas"] = ventas
    st.session_state["_inventario"] = inventario

    st.sidebar.header("Parametros")
    ventana = st.sidebar.number_input(
        "Ventana anual (turnos)", min_value=13, max_value=200,
        value=VENTANA_ANUAL_DEFAULT, step=1,
        help="Turnos recientes usados para el ABC (por defecto 52 ~ 1 ano).",
    )
    n_windows = st.sidebar.number_input(
        "n_windows (validacion cruzada)", min_value=1, max_value=20,
        value=N_WINDOWS_DEFAULT, step=1,
    )
    h = st.sidebar.number_input(
        "h (periodos a pronosticar)", min_value=1, max_value=24,
        value=H_DEFAULT, step=1,
    )

    # --- Resumen de carga y validacion (siempre visible, arriba de las
    #     pestanas: es el "gate" antes de continuar).
    validacion = validar_datos(demanda, costos)
    n_sku = demanda.groupby(["REGIONAL", "PRODUCTO"]).ngroups
    turn_min = int(demanda["Turn"].min())
    turn_max = int(demanda["Turn"].max())

    st.subheader("Resumen de carga")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Archivo", archivo.name)
    col2.metric("SKU validos", n_sku)
    col3.metric("Rango de Turn", f"{turn_min} - {turn_max}")
    col4.metric("Ventana anual", f"{ventana} turnos")

    if validacion["ok"]:
        st.success("Validacion de datos: correcta. Navegue las pestanas siguientes.")
    else:
        st.error("La validacion de datos fallo. Revise los controles:")
        for descripcion, estado in validacion["controles"]:
            if estado:
                st.write(f"OK   {descripcion}")
            else:
                st.write(f"FALLA   {descripcion}")
        st.warning("No se puede continuar hasta corregir los datos. "
                   "La aplicacion no corrige datos silenciosamente.")
        return

    # --- Calculos previos a las pestanas (ABC y motor cacheado).
    abc, info_ventana = calcular_abc(demanda, costos, ventana)
    comparacion, ganadores, pronosticos, errores, df_ts = _motor_cacheado(
        demanda, n_windows, h
    )

    # --- Cálculo de indicadores adicionales (IFR, rotación, GMROI).
    # Se calculan solo si existen datos de ventas e inventario.
    ifr_df = pd.DataFrame()
    ifr_global = 1.0
    ifr_por_ref = {}
    rot_df = pd.DataFrame()
    gmroi_df = pd.DataFrame()

    if not ventas.empty and not inventario.empty:
        try:
            # IFR semanal
            ifr_df, ifr_global, ifr_por_ref = preparar_datos_ifr(ventas, inventario)

            # Rotación y GMROI por referencia
            rot_por_regional = []
            for regional in ["NORTE", "CENTRO", "SUR"]:
                rot = calcular_inv_turnover(ventas, inventario, regional)
                if not rot.empty:
                    rot_por_regional.append(rot)
                gmroi = calcular_gmroi(ventas, inventario, costos, regional)
                if not gmroi.empty:
                    gmroi_df = pd.concat(
                        [gmroi_df, gmroi], ignore_index=True
                    )

            if rot_por_regional:
                rot_df = pd.concat(rot_por_regional, ignore_index=True)

        except Exception as e:
            st.warning(f"No se pudieron calcular todos los indicadores: {e}")


    # Defaults de umbrales XYZ a partir de la distribucion de Score%.
    scores_validos = [
        v.get("Score%", np.nan) for v in ganadores.values()
        if not pd.isna(v.get("Score%", np.nan))
    ]
    if scores_validos:
        default_xy = float(np.percentile(scores_validos, 33))
        default_yz = float(np.percentile(scores_validos, 67))
    else:
        default_xy, default_yz = 0.30, 0.60

    # --- Pestanas: una por seccion (secuencia de la seccion 18).
    tab_carga, tab_costos, tab_abc, tab_pron, tab_xyz, tab_abcxyz, \
        tab_cons, tab_rs, tab_ifr, tab_rotgmroi, tab_concl = st.tabs([
            "1. Carga y validacion",
            "2. Costos y precios",
            "3. ABC",
            "4. Pronosticos",
            "5. XYZ",
            "6. ABC-XYZ",
            "7. Consolidado",
            "8. Revision periodica (R,S)",
            "9. IFR y Rotacion de Inventarios (indicadores de servicio)",
            "10. Inventarios y GMROI",
            "11. Conclusiones y exportacion",
        ])

    # --- Pestana 1: detalle de la validacion.
    with tab_carga:
        st.markdown("#### Controles de validacion de datos")
        for descripcion, estado in validacion["controles"]:
            if estado:
                st.write(f"OK   {descripcion}")
            else:
                st.write(f"FALLA   {descripcion}")
        st.caption(
            "Ventana anual movil para ABC: ultimos "
            f"{ventana} turnos. Un turno = un periodo."
        )

    # --- Pestana 2: costos y precios.
    with tab_costos:
        st.markdown("#### Costos y precios")
        st.markdown("Costo unitario de fabricacion = **MD + MOD** "
                    "(sin transporte ni almacenamiento).")
        costos_show = costos.copy()
        costos_show["Utilidad unitaria"] = (
            costos_show["Precio"].astype(float)
            - costos_show["Costo"].astype(float)
        )
        st.dataframe(costos_show, use_container_width=True)

    # --- Pestana 3: ABC.
    with tab_abc:
        st.markdown("#### Clasificacion ABC (por Utilidad total)")
        st.caption(
            f"Ventana anual: Turn {info_ventana['turn_inicial']} a "
            f"{info_ventana['turn_final']} "
            f"({info_ventana['observaciones']} observaciones). "
            "A: acumulado <= 80%; B: > 80% y <= 95%; C: > 95%."
        )
        abc_display = abc.copy()
        abc_display["Participacion %"] = abc_display["Participacion"] * 100
        abc_display["Acumulado %"] = abc_display["Acumulado"] * 100
        st.dataframe(
            abc_display.drop(columns=["Participacion", "Acumulado"]),
            use_container_width=True,
        )
        st.plotly_chart(grafica_pareto(abc), use_container_width=True)

    # --- Pestana 4: pronosticos.
    with tab_pron:
        st.markdown("#### Motor de pronosticos")
        if errores:
            with st.expander(f"Advertencias del motor ({len(errores)})"):
                for err in errores:
                    st.warning(err)
        st.markdown("**Comparacion de modelos (Score%)**")
        if not comparacion.empty:
            comp_display = comparacion.copy()
            for c in ["MAE%", "Sesgo%", "Score%"]:
                comp_display[c] = comp_display[c] * 100
            st.dataframe(comp_display, use_container_width=True)
        else:
            st.warning("No se genero comparacion de modelos.")
        st.markdown(
            "**Pronosticos por region y agregados.** "
            "Linea continua = real; punteada = modelo ganador."
        )
        st.plotly_chart(
            grafica_pronosticos_oficial(df_ts, pronosticos, h, turn_max),
            use_container_width=True,
        )


    # --- Pestana 5: XYZ (cortes configurables y visibles).
    with tab_xyz:
        st.markdown("#### Clasificacion XYZ (por Score% del ganador)")
        if scores_validos:
            st.markdown(
                f"Distribucion de Score% de los 8 SKU: "
                f"min={min(scores_validos):.2%}, "
                f"mediana={np.median(scores_validos):.2%}, "
                f"max={max(scores_validos):.2%}."
            )
        c1, c2 = st.columns(2)
        limite_xy = c1.number_input(
            "Limite X/Y (Score%)", min_value=0.0, max_value=1.0,
            value=float(default_xy), step=0.01, format="%.2f",
        )
        limite_yz = c2.number_input(
            "Limite Y/Z (Score%)", min_value=0.0, max_value=1.0,
            value=float(default_yz), step=0.01, format="%.2f",
        )
        st.caption("Regla: Score% <= X/Y -> X; X/Y < Score% <= Y/Z -> Y; "
                   "Score% > Y/Z -> Z. Umbrales visibles y exportados "
                   "(no inventados).")

    # Los cortes se validan igual para poder construir ABC-XYZ.
    umbral_ok = limite_xy < limite_yz
    if not umbral_ok:
        st.error("Umbrales inconsistentes: el limite X/Y debe ser menor "
                 "que Y/Z. Corrija los limites en la pestana XYZ.")

    xyz = clasificar_xyz(ganadores, limite_xy, limite_yz) if umbral_ok else {}

    # --- Pestana 6: ABC-XYZ.
    with tab_abcxyz:
        st.markdown("#### Clasificacion ABC-XYZ (8 SKU)")
        if umbral_ok:
            abc_xyz = combinar_abc_xyz(abc, ganadores, xyz)
            st.dataframe(abc_xyz, use_container_width=True)
            n_filas = len(abc_xyz)
            if n_filas != 8:
                st.error(f"Se esperaban exactamente 8 SKU, pero hay {n_filas}.")
            else:
                st.success("ABC-XYZ: exactamente 8 SKU "
                           "(CEDI/MOTOTRAK excluidos).")
            cc1, cc2 = st.columns(2)
            cc1.plotly_chart(grafica_score(abc_xyz), use_container_width=True)
            cc2.plotly_chart(grafica_matriz_abcxyz(abc_xyz),
                             use_container_width=True)
        else:
            abc_xyz = pd.DataFrame()
            st.info("Ajuste los limites XYZ validos para ver ABC-XYZ.")

    # --- Pestana 7: consolidado por producto.
    with tab_cons:
        st.markdown("#### Consolidado por producto")
        st.caption("Demanda y utilidad consolidadas de Moto, Cuatrimoto y "
                   "Tractor. No altera el ABC-XYZ de los 8 SKU.")
        cons = consolidado_producto(demanda, costos, ventana)
        st.dataframe(cons, use_container_width=True)

    # --- Pestana 8: modelo de inventarios de revision periodica (R, S).
    with tab_rs:
        st.markdown("#### Modelo de inventarios: revision periodica (R, S)")
        st.markdown(
            "Cada **R** turnos se revisa el inventario y se reordena hasta el "
            "nivel objetivo **S**. Se usan los **pronosticos** del modelo "
            "ganador y el **RMSE** del motor (no el promedio ni la desviacion "
            "del historico) para estimar la demanda del periodo de proteccion "
            "y el stock de seguridad.")

    # --- Pestana 9: Indicador de Tasa de Llenado (IFR)
    with tab_ifr:
        st.markdown("#### Tasa de llenado por producto y referencia (IFR)")
        st.markdown(
            "El **IFR (Item Fill Rate)** mide el porcentaje de demanda "
            "cubierta por la disponibilidad de inventario. Se calcula "
            "semanalmente para cada producto en cada referencia."
        )

        if ifr_df.empty:
            st.info("No hay datos suficientes de ventas e inventario para "
                    "calcular el IFR semanal.")
        else:
            st.plotly_chart(grafica_ifr_semanal_por_referencia(ifr_df),
                            use_container_width=True)
            st.dataframe(ifr_df, use_container_width=True)

            # IFR global
            st.subheader("IFR Global")
            st.metric("IFR Global", f"{ifr_global * 100:.2f}%")
            st.plotly_chart(
                grafica_ifr_global(ifr_global, ifr_por_ref),
                use_container_width=True
            )
            st.caption("IFR Global = unidades cubiertas / unidades demanadas "
                       "en el periodo.")

    # --- Pestana 10: Rotacion de inventarios y GMROI
    with tab_rotgmroi:
        st.markdown("#### Rotacion de inventarios y GMROI por regional")
        st.markdown(
            "La **rotacion de inventarios (ROT)** mide cuantas veces el "
            "inventario se 'gira' durante un periodo: ROT = Demanda / "
            "Inventario."
        )
        st.markdown(
            "El **GMROI (Gross Margin Return On Investment)** mide la "
            "utilidad generada por cada peso invertido en inventario: "
            "GMROI = Utilidad bruta / Costo de inventario."
        )

        if rot_df.empty:
            st.info("No hay datos suficientes de ventas e inventario para "
                    "calcular la rotacion.")
        else:
            st.subheader("Rotacion de Inventarios")
            st.plotly_chart(grafica_rotacion_inventarios(rot_df),
                            use_container_width=True)
            st.dataframe(rot_df, use_container_width=True)

        if not gmroi_df.empty:
            st.subheader("GMROI (Gross Margin Return On Investment)")
            st.plotly_chart(grafica_gmroi(gmroi_df), use_container_width=True)
            st.dataframe(gmroi_df, use_container_width=True)
        elif rot_df.empty:
            st.info("No hay datos suficientes de ventas, inventario y costos "
                    "para calcular el GMROI.")

        rs1, rs2, rs3 = st.columns(3)
        R = rs1.number_input("R (periodo de revision, turnos)",
                             min_value=1, max_value=20, value=4, step=1)
        L = rs2.number_input("L (lead time / entrega, turnos)",
                             min_value=0, max_value=20, value=2, step=1)
        NC = rs3.number_input("NC (nivel de servicio, %)",
                              min_value=50.0, max_value=99.99, value=95.0,
                              step=0.1, format="%.2f",
                              help="Nivel de servicio objetivo en %. El "
                                   "modelo lo convierte internamente al "
                                   "factor Z = Phi^-1(NC/100) y usa ese Z "
                                   "para el stock de seguridad (SS = Z*sigma_P). "
                                   "Ej.: 90% -> Z=1.282; 95% -> Z=1.645; "
                                   "98% -> Z=2.054.")
        # El usuario fija NC en % (valor que cambia a voluntad); internamente
        # se sigue usando el factor Z, igual que antes, para calcular SS.
        Z = float(norm.ppf(NC / 100.0))

        P_rs = int(R) + int(L)
        if P_rs > h:
            st.warning(
                f"El periodo de proteccion ({P_rs} turnos) supera el "
                f"horizonte pronosticado ({h} turnos). Aumente 'h' en la "
                "barra lateral para cubrir la demanda con los pronosticos."
            )

        # --- Tabla 1: salida del modelo (R,S) en solo lectura. Los insumos
        # del juego (I y IO) y la cantidad a pedir Q viven en la Tabla 2.
        # Q (cantidad a pedir) = S - (Inventario disponible I + Ordenes en
        # transito IO). Los valores de I e IO los toma el usuario del juego
        # V-logistics en cada corrida y se guardan para no perderlos.
        rs_tabla = disenar_revision_periodica(ganadores, pronosticos,
                                              R, L, Z)
        rs_tabla["Ref"] = (rs_tabla["Producto"] + "-"
                           + rs_tabla["Regional"])

        # Estado persistente de los insumos del juego por referencia.
        if "rs_inputs" not in st.session_state:
            st.session_state["rs_inputs"] = {}  # {Ref: {"I":.., "IO":..}}

        def _num(valor):
            """Convierte un valor editado a float.

            Vacío / None / NaN  -> np.nan  (desconocido: no se asume cero).
            Cero explícito     -> 0.0     (confirmado por el usuario).
            """
            if valor is None:
                return np.nan
            if isinstance(valor, float) and np.isnan(valor):
                return np.nan
            try:
                f = float(valor)
            except (TypeError, ValueError):
                return np.nan
            return f

        st.caption(
            "Tabla 1: una fila por referencia (Norte, Centro, Sur y CEDI; Sur "
            "excluye Tractor-Sur). MOTOTRAK no se incluye. Los insumos del "
            "juego (I y IO) y la cantidad a pedir Q se manejan en la Tabla 2."
        )
        columnas_t1 = [
            "Ref", "Regional", "Producto", "Modelo ganador", "P = R+L",
            "Demanda en P (suma pronosticos)", "RMSE (motor)",
            "sigma_P = RMSE*sqrt(P)",
            "Stock de seguridad", "Nivel objetivo S",
        ]
        st.dataframe(rs_tabla[columnas_t1], use_container_width=True,
                     hide_index=True)

        # --- Indicacion 2: control de calidad del dato. Una serie con
        # RMSE ~ 0 se ajusta de forma determinista (valor constante o captura
        # erronea, p. ej. demanda = numero de turno); el modelo no pone stock
        # de seguridad y su S/Q puede ser un artefacto que sobreinfla el
        # pedido. Se marca para verificar el Excel fuente antes de pedir.
        sospechosas = rs_tabla[
            rs_tabla["RMSE (motor)"].notna()
            & (rs_tabla["RMSE (motor)"] < 0.01)
        ]
        if not sospechosas.empty:
            refs_sospecha = ", ".join(sospechosas["Ref"].tolist())
            st.warning(
                f"**Calidad del dato (Indicacion 2):** {len(sospechosas)} "
                f"referencia(s) con RMSE ~ 0: {refs_sospecha}. Una serie con "
                "RMSE casi nulo es determinista (constante o captura erronea, "
                "p. ej. demanda igual al numero de turno); el modelo no aplica "
                "stock de seguridad y su S/Q puede ser un artefacto que "
                "sobreinfla el pedido. Verifique esas columnas en el Excel "
                "fuente antes de dar luz verde."
            )

        # --- Tabla 2: insumos editables (I y IO) y Q ya recalculado.
        # Las ediciones del navegador quedan pendientes al inicio del rerun
        # (st.session_state conserva el estado del data_editor). Se aplican
        # ANTES de construir la tabla para que la columna Q salga actualizada
        # en la misma corrida en que se modifico I u IO. Las posiciones de
        # fila solo valen si el orden de referencias no cambio, por eso se
        # compara con rs_refs (guardado al renderizar la corrida anterior).
        refs_actuales = list(rs_tabla["Ref"])
        pendientes = {}
        if st.session_state.get("rs_refs") == refs_actuales:
            estado = st.session_state.get("editor_rs2")
            if isinstance(estado, dict):
                for pos, cols in (estado.get("edited_rows") or {}).items():
                    try:
                        i = int(pos)
                    except (TypeError, ValueError):
                        continue
                    if (not isinstance(cols, dict)
                            or not (0 <= i < len(refs_actuales))):
                        continue
                    fila = pendientes.setdefault(refs_actuales[i], {})
                    if "Inventario disponible (I)" in cols:
                        fila["I"] = cols["Inventario disponible (I)"]
                    if "Ordenes en transito (IO)" in cols:
                        fila["IO"] = cols["Ordenes en transito (IO)"]

        def _insumo(ref, key):
            """I/IO vigente: edicion pendiente si existe, si no rs_inputs."""
            if key in pendientes.get(ref, {}):
                return _num(pendientes[ref][key])
            return _num(
                st.session_state["rs_inputs"].get(ref, {}).get(key, 0.0)
            )

        def _estado(desv, s_obj):
            """Clasifica la posicion de inventario vs el nivel objetivo S."""
            if pd.isna(desv) or pd.isna(s_obj):
                return "Sin dato"
            if desv > 0:
                return "Sobrestock"
            if desv >= -0.10 * s_obj:
                return "Al limite"
            return "En riesgo de quiebre"

        st.markdown("**Q: cantidad a pedir**")
        st.caption(
            "Edite el **Inventario disponible (I)** y las **Ordenes en "
            "transito (IO)** observados en el juego; **Q (cantidad a pedir)** "
            "y el **Estado** se recalculan al instante. Desviacion = "
            "(I + IO) - S: >0 sobrestock (congele su Q); 0 a -10% de S al "
            "limite; < -10% de S en riesgo de quiebre (luz verde urgente)."
        )
        _i_vals = [_insumo(r, "I") for r in rs_tabla["Ref"]]
        _io_vals = [_insumo(r, "IO") for r in rs_tabla["Ref"]]
        _pos_vals = [i + io for i, io in zip(_i_vals, _io_vals)]
        _s_vals = rs_tabla["Nivel objetivo S"]
        _desv_vals = [
            (p - s) if pd.notna(s) else np.nan
            for p, s in zip(_pos_vals, _s_vals)
        ]
        _est_vals = [_estado(d, s) for d, s in zip(_desv_vals, _s_vals)]
        _pos_series = pd.Series(_pos_vals, index=rs_tabla.index)
        _q_vals = np.where(
            _s_vals.notna() & _pos_series.notna(),
            (_s_vals - _pos_series).clip(lower=0),
            np.nan,
        )
        tabla2 = pd.DataFrame({
            "Ref": rs_tabla["Ref"],
            "Estado": _est_vals,
            "Nivel objetivo S": _s_vals,
            "Inventario disponible (I)": _i_vals,
            "Ordenes en transito (IO)": _io_vals,
            "Posicion de inventario (I+IO)": _pos_vals,
            "Desviacion vs S (I+IO - S)": _desv_vals,
            "Q (cantidad a pedir)": _q_vals,
        })
        edited = st.data_editor(
            tabla2,
            column_config={
                "Inventario disponible (I)": st.column_config.NumberColumn(
                    "Inventario disponible (I)", min_value=0, step=1,
                    help="Unidades fisicas disponibles (del juego)."),
                "Ordenes en transito (IO)": st.column_config.NumberColumn(
                    "Ordenes en transito (IO)", min_value=0, step=1,
                    help="Unidades ya pedidas y en transito (del juego)."),
            },
            disabled=[
                "Ref",
                "Estado",
                "Nivel objetivo S",
                "Posicion de inventario (I+IO)",
                "Desviacion vs S (I+IO - S)",
                "Q (cantidad a pedir)",
            ],
            use_container_width=True,
            hide_index=True,
            key="editor_rs2",
        )

        # Guardamos los insumos editados para no perderlos entre corridas.
        for _, fila in edited.iterrows():
            st.session_state["rs_inputs"][fila["Ref"]] = {
                "I": _num(fila["Inventario disponible (I)"]),
                "IO": _num(fila["Ordenes en transito (IO)"]),
            }
        st.session_state["rs_refs"] = list(edited["Ref"])

        # rs_tabla completa (modelo + insumos + estado + Q) para grafico y exportacion.
        mapa_ed = edited.set_index("Ref")
        rs_tabla["Inventario disponible (I)"] = rs_tabla["Ref"].map(
            mapa_ed["Inventario disponible (I)"])
        rs_tabla["Ordenes en transito (IO)"] = rs_tabla["Ref"].map(
            mapa_ed["Ordenes en transito (IO)"])
        rs_tabla["Posicion de inventario (I+IO)"] = (
            rs_tabla["Inventario disponible (I)"]
            + rs_tabla["Ordenes en transito (IO)"]
        )
        rs_tabla["Desviacion vs S (I+IO - S)"] = (
            rs_tabla["Posicion de inventario (I+IO)"]
            - rs_tabla["Nivel objetivo S"]
        )
        rs_tabla["Estado"] = [
            _estado(d, s) for d, s in zip(
                rs_tabla["Desviacion vs S (I+IO - S)"],
                rs_tabla["Nivel objetivo S"])
        ]
        rs_tabla["Q (cantidad a pedir)"] = np.where(
            rs_tabla["Nivel objetivo S"].notna()
            & rs_tabla["Posicion de inventario (I+IO)"].notna(),
            (rs_tabla["Nivel objetivo S"]
             - rs_tabla["Posicion de inventario (I+IO)"]).clip(lower=0),
            np.nan,
        )

        total_q = float(np.nansum(rs_tabla["Q (cantidad a pedir)"].values))
        st.info(f"Cantidad total a pedir en esta corrida (suma de Q): "
                f"{total_q:,.0f} unidades.")

        # Grafico comparativo: nivel objetivo S vs cantidad a pedir Q.
        rs_ok = rs_tabla.dropna(subset=["Nivel objetivo S"]).copy()
        if not rs_ok.empty and "Q (cantidad a pedir)" in rs_ok.columns:
            import plotly.express as px
            fig_rs = px.bar(
                rs_ok, x="Ref", y=["Nivel objetivo S", "Q (cantidad a pedir)"],
                barmode="group",
                title="Nivel objetivo S vs cantidad a pedir Q por referencia",
                template="ggplot2", height=430,
            )
            st.plotly_chart(fig_rs, use_container_width=True)
        elif not rs_ok.empty:
            import plotly.express as px
            fig_rs = px.bar(
                rs_ok, x="Ref", y="Nivel objetivo S", color="Regional",
                title="Nivel objetivo S por referencia (revision periodica)",
                template="ggplot2", height=420,
            )
            st.plotly_chart(fig_rs, use_container_width=True)



    # --- Pestana 8: conclusiones y exportacion.
    with tab_concl:
        st.markdown("#### Conclusiones y exportacion")
        st.markdown("Documente al menos cuatro conclusiones/observaciones "
                    "(Hallazgo, Interpretacion, Recomendacion).")
        if "conclusiones" not in st.session_state:
            st.session_state["conclusiones"] = ["", "", "", ""]
        for i in range(len(st.session_state["conclusiones"])):
            st.session_state["conclusiones"][i] = st.text_area(
                f"Conclusion / Observacion {i + 1}",
                value=st.session_state["conclusiones"][i],
                key=f"concl_{i}", height=80,
            )
        if st.button("Agregar otra conclusion"):
            st.session_state["conclusiones"].append("")
            st.experimental_rerun()

        conclusiones_validas = [
            c for c in st.session_state["conclusiones"] if c.strip()
        ]
        st.info(f"Conclusiones solidas registradas: {len(conclusiones_validas)} "
                f"(minimo recomendado: 4).")

        st.markdown("---")
        nombre_salida = "Resultados_ABC_XYZ_Mototrack.xlsx"
        ruta_salida = os.path.join(os.getcwd(), nombre_salida)
        listo_exportar = umbral_ok and not abc_xyz.empty
        if not listo_exportar:
            st.warning("Corrija los limites XYZ (pestana 5) para habilitar la "
                       "exportacion con la clasificacion ABC-XYZ completa.")
        if st.button("Generar Excel de resultados", disabled=not listo_exportar):
            try:
                generar_excel(
                    abc, costos, abc_xyz, comparacion, pronosticos,
                    info_ventana, n_windows, h, limite_xy, limite_yz,
                    conclusiones_validas if conclusiones_validas
                    else ["(sin conclusiones)"],
                    ruta_salida,
                    rs_tabla=rs_tabla,
                    ventas=ventas,
                    inventario=inventario,
                )
                st.success(f"Excel generado: {nombre_salida} "
                           "(en la carpeta del proyecto).")
                with open(ruta_salida, "rb") as f:
                    st.download_button(
                        "Descargar Excel", data=f.read(),
                        file_name=nombre_salida,
                        mime="application/vnd.openxmlformats-officedocument."
                        "spreadsheetml.sheet",
                    )
            except Exception as e:
                st.error(f"Error al generar el Excel: {e}")


if __name__ == "__main__":
    main()










