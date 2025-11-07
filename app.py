import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Configuración de página
st.set_page_config(
    page_title="RRHH Analytics Pro",
    page_icon="🏗️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Estilos CSS personalizados
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: bold;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 15px;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .alert-high {
        background: linear-gradient(135deg, #ff6b6b 0%, #ee5a52 100%);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
    }
    .alert-medium {
        background: linear-gradient(135deg, #ffd93d 0%, #ffcd3c 100%);
        color: black;
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
    }
    .section-header {
        font-size: 1.8rem;
        color: #2c3e50;
        border-bottom: 3px solid #3498db;
        padding-bottom: 0.5rem;
        margin-bottom: 1.5rem;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    """Generar datos sintéticos completos para la demo"""
    np.random.seed(42)
    
    # Generar empleados
    nombres = ['Sofia', 'Martina', 'Lucia', 'Ana', 'Carolina', 'Valentina', 
               'Carlos', 'Diego', 'Juan', 'Pablo', 'Ricardo', 'Javier', 
               'Miguel', 'Roberto', 'Fernando']
    apellidos = ['Lopez', 'Gonzalez', 'Garcia', 'Martinez', 'Rodriguez', 
                 'Perez', 'Diaz', 'Gomez', 'Fernandez', 'Romero']
    
    empleados = []
    for i in range(150):
        genero = np.random.choice(['Femenino', 'Masculino'], p=[0.35, 0.65])
        dept = np.random.choice(['Albañilería', 'Electricidad', 'Plomería', 'Herrería', 'Pintura'], 
                               p=[0.3, 0.25, 0.2, 0.15, 0.1])
        
        # Salarios base por departamento
        salario_base = {
            'Albañilería': 80000, 'Electricidad': 95000, 
            'Plomería': 85000, 'Herrería': 110000, 'Pintura': 75000
        }[dept]
        
        salario = salario_base * np.random.uniform(0.8, 1.5)
        experiencia = np.random.randint(6, 120)
        
        empleados.append({
            'id': f"EMP{i+1:03d}",
            'nombre': np.random.choice(nombres),
            'apellido': np.random.choice(apellidos),
            'genero': genero,
            'edad': np.random.randint(22, 60),
            'departamento': dept,
            'cargo': f"{dept} {'Senior' if experiencia > 60 else 'Junior'}",
            'salario': round(salario, 2),
            'fecha_contratacion': datetime.now() - timedelta(days=np.random.randint(30, 365*3)),
            'experiencia_meses': experiencia,
            'ubicacion': np.random.choice(['Sede Central', 'Obra Norte', 'Obra Sur', 'Obra Este']),
            'nivel_educacion': np.random.choice(['Secundario', 'Terciario', 'Universitario', 'Maestría'], 
                                              p=[0.4, 0.3, 0.2, 0.1]),
            'activo': np.random.choice([True, False], p=[0.92, 0.08])
        })
    
    df_empleados = pd.DataFrame(empleados)
    
    # Generar obras
    obras = []
    for i in range(12):
        obras.append({
            'id': f"OBR{i+1:03d}",
            'nombre': f"Proyecto {['Residencial', 'Comercial', 'Industrial'][i % 3]} {i+1}",
            'ubicacion': np.random.choice(['Nordelta', 'Pilar', 'Tigre', 'Escobar', 'San Isidro']),
            'presupuesto': np.random.randint(5000000, 25000000),
            'fecha_inicio': datetime.now() - timedelta(days=np.random.randint(30, 300)),
            'duracion_estimada': np.random.randint(90, 360),
            'estado': np.random.choice(['En Planificación', 'En Progreso', 'En Riesgo', 'Completado'], 
                                     p=[0.1, 0.6, 0.2, 0.1]),
            'gerente': np.random.choice([f"{emp['nombre']} {emp['apellido']}" for emp in empleados[:20]]),
            'complejidad': np.random.choice(['Baja', 'Media', 'Alta'], p=[0.3, 0.5, 0.2])
        })
    
    df_obras = pd.DataFrame(obras)
    
    # Generar asistencias y rendimiento
    asistencias = []
    for _ in range(2000):
        emp = empleados[np.random.randint(0, len(empleados))]
        obra = obras[np.random.randint(0, len(obras))]
        
        fecha = datetime.now() - timedelta(days=np.random.randint(1, 90))
        
        asistencias.append({
            'empleado_id': emp['id'],
            'obra_id': obra['id'],
            'fecha': fecha,
            'horas_trabajadas': np.random.randint(6, 10),
            'horas_extra': np.random.choice([0, 0, 0, 1, 2]),  # 40% probabilidad de horas extra
            'productividad': np.random.normal(85, 10),
            'calidad_trabajo': np.random.normal(90, 5),
            'incidentes_seguridad': np.random.poisson(0.1),
            'ausente': np.random.choice([True, False], p=[0.05, 0.95])
        })
    
    df_asistencias = pd.DataFrame(asistencias)
    df_asistencias['productividad'] = df_asistencias['productividad'].clip(50, 100)
    df_asistencias['calidad_trabajo'] = df_asistencias['calidad_trabajo'].clip(70, 100)
    
    return df_empleados, df_obras, df_asistencias

def main():
    # Header principal
    st.markdown('<h1 class="main-header">🏗️ RRHH Analytics Pro</h1>', unsafe_allow_html=True)
    
    # Cargar datos
    df_empleados, df_obras, df_asistencias = load_data()
    
    # Sidebar - Navegación
    st.sidebar.image("https://via.placeholder.com/200x50/1f77b4/ffffff?text=RRHH+PRO", use_column_width=True)
    st.sidebar.title("Navegación")
    
    menu = st.sidebar.radio(
        "Seleccione módulo:",
        ["📊 Dashboard Ejecutivo", "👥 Gestión de Personal", "🏗️ Gestión de Obras", 
         "📈 Analytics Avanzado", "🎯 Desempeño", "⚠️ Alertas Tempranas", "⚙️ Configuración"]
    )
    
    # KPIs Principales - Siempre visibles
    st.markdown("### 📈 Métricas Clave en Tiempo Real")
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        total_empleados = len(df_empleados[df_empleados['activo']])
        st.metric("👥 Empleados Activos", total_empleados, delta="+5%")
    
    with col2:
        costo_total = df_empleados[df_empleados['activo']]['salario'].sum()
        st.metric("💰 Costo Mensual", f"${costo_total:,.0f}")
    
    with col3:
        productividad_promedio = df_asistencias['productividad'].mean()
        st.metric("📊 Productividad", f"{productividad_promedio:.1f}%", delta="+2.1%")
    
    with col4:
        rotacion = len(df_empleados[~df_empleados['activo']]) / len(df_empleados) * 100
        st.metric("🔄 Rotación", f"{rotacion:.1f}%", delta="-1.2%", delta_color="inverse")
    
    with col5:
        obras_activas = len(df_obras[df_obras['estado'] == 'En Progreso'])
        st.metric("🏗️ Obras Activas", obras_activas)
    
    st.markdown("---")
    
    # Contenido según menú seleccionado
    if menu == "📊 Dashboard Ejecutivo":
        show_executive_dashboard(df_empleados, df_obras, df_asistencias)
    elif menu == "👥 Gestión de Personal":
        show_person_management(df_empleados, df_asistencias)
    elif menu == "🏗️ Gestión de Obras":
        show_project_management(df_obras, df_asistencias, df_empleados)
    elif menu == "📈 Analytics Avanzado":
        show_advanced_analytics(df_empleados, df_asistencias)
    elif menu == "🎯 Desempeño":
        show_performance_analytics(df_empleados, df_asistencias)
    elif menu == "⚠️ Alertas Tempranas":
        show_early_warnings(df_empleados, df_obras, df_asistencias)
    elif menu == "⚙️ Configuración":
        show_configuration()

def show_executive_dashboard(df_empleados, df_obras, df_asistencias):
    st.markdown('<div class="section-header">📊 Dashboard Ejecutivo</div>', unsafe_allow_html=True)
    
    # Primera fila de gráficos
    col1, col2, col3 = st.columns([2, 1, 1])
    
    with col1:
        # Mapa de calor de productividad por departamento y ubicación
        st.subheader("🌡️ Mapa de Calor - Productividad")
        heatmap_data = df_empleados.merge(df_asistencias, left_on='id', right_on='empleado_id')
        pivot_table = heatmap_data.pivot_table(
            values='productividad', 
            index='departamento', 
            columns='ubicacion', 
            aggfunc='mean'
        ).fillna(0)
        
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.heatmap(pivot_table, annot=True, cmap='RdYlGn', center=85, ax=ax)
        plt.title('Productividad Promedio por Departamento y Ubicación')
        st.pyplot(fig)
    
    with col2:
        # Distribución salarial
        st.subheader("💰 Distribución Salarial")
        fig = px.box(df_empleados, x='departamento', y='salario', color='departamento')
        fig.update_layout(showlegend=False, height=300)
        st.plotly_chart(fig, use_container_width=True)
        
        # Tasa de rotación por departamento
        st.subheader("🔄 Rotación por Depto.")
        rotacion_depto = df_empleados.groupby('departamento')['activo'].apply(
            lambda x: (1 - x.mean()) * 100
        ).reset_index()
        fig = px.bar(rotacion_depto, x='departamento', y='activo')
        fig.update_layout(height=250)
        st.plotly_chart(fig, use_container_width=True)
    
    with col3:
        # Métricas rápidas
        st.subheader("📋 Métricas Rápidas")
        
        avg_salary = df_empleados['salario'].mean()
        avg_experience = df_empleados['experiencia_meses'].mean()
        avg_age = df_empleados['edad'].mean()
        
        st.metric("Edad Promedio", f"{avg_age:.1f} años")
        st.metric("Experiencia Promedio", f"{avg_experience/12:.1f} años")
        st.metric("Salario Promedio", f"${avg_salary:,.0f}")
        st.metric("Horas Extra/Mes", "45h", "+8%")
    
    # Segunda fila - Análisis temporal
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📈 Tendencia de Productividad (Últimos 3 meses)")
        df_asistencias['fecha'] = pd.to_datetime(df_asistencias['fecha'])
        df_asistencias['semana'] = df_asistencias['fecha'].dt.isocalendar().week
        
        productividad_semanal = df_asistencias.groupby('semana')['productividad'].mean().reset_index()
        fig = px.line(productividad_semanal, x='semana', y='productividad', 
                     title='Evolución Semanal de Productividad')
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("🎯 Distribución de Competencias")
        fig = px.pie(df_empleados, names='nivel_educacion', 
                    title='Distribución por Nivel Educativo')
        st.plotly_chart(fig, use_container_width=True)

def show_person_management(df_empleados, df_asistencias):
    st.markdown('<div class="section-header">👥 Gestión de Personal</div>', unsafe_allow_html=True)
    
    # Filtros avanzados
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        dept_filter = st.multiselect(
            "Departamento",
            options=df_empleados['departamento'].unique(),
            default=df_empleados['departamento'].unique()
        )
    
    with col2:
        ubicacion_filter = st.multiselect(
            "Ubicación",
            options=df_empleados['ubicacion'].unique(),
            default=df_empleados['ubicacion'].unique()
        )
    
    with col3:
        educacion_filter = st.multiselect(
            "Nivel Educativo",
            options=df_empleados['nivel_educacion'].unique(),
            default=df_empleados['nivel_educacion'].unique()
        )
    
    with col4:
        estado_filter = st.multiselect(
            "Estado",
            options=['Activo', 'Inactivo'],
            default=['Activo']
        )
    
    # Aplicar filtros
    mask = (
        df_empleados['departamento'].isin(dept_filter) &
        df_empleados['ubicacion'].isin(ubicacion_filter) &
        df_empleados['nivel_educacion'].isin(educacion_filter) &
        df_empleados['activo'].isin([True if 'Activo' in estado_filter else False, 
                                   False if 'Inactivo' in estado_filter else True])
    )
    
    empleados_filtrados = df_empleados[mask]
    
    st.metric("Empleados Filtrados", len(empleados_filtrados))
    
    # Visualización de datos
    col1, col2 = st.columns([3, 2])
    
    with col1:
        st.subheader("📋 Lista de Empleados")
        
        # Mostrar métricas de los filtrados
        if len(empleados_filtrados) > 0:
            col_metrics1, col_metrics2, col_metrics3 = st.columns(3)
            with col_metrics1:
                st.metric("Salario Promedio", f"${empleados_filtrados['salario'].mean():,.0f}")
            with col_metrics2:
                st.metric("Experiencia Promedio", f"{empleados_filtrados['experiencia_meses'].mean()/12:.1f} años")
            with col_metrics3:
                st.metric("Edad Promedio", f"{empleados_filtrados['edad'].mean():.1f} años")
        
        st.dataframe(
            empleados_filtrados[[
                'id', 'nombre', 'apellido', 'departamento', 'cargo', 
                'salario', 'experiencia_meses', 'ubicacion'
            ]].rename(columns={
                'id': 'ID', 'nombre': 'Nombre', 'apellido': 'Apellido',
                'departamento': 'Departamento', 'cargo': 'Cargo',
                'salario': 'Salario', 'experiencia_meses': 'Exp (meses)',
                'ubicacion': 'Ubicación'
            }),
            use_container_width=True,
            height=400
        )
    
    with col2:
        st.subheader("📊 Análisis del Personal")
        
        # Gráfico de distribución por departamento
        fig = px.pie(empleados_filtrados, names='departamento', 
                    title='Distribución por Departamento')
        st.plotly_chart(fig, use_container_width=True)
        
        # Gráfico de dispersión salario vs experiencia
        fig = px.scatter(empleados_filtrados, x='experiencia_meses', y='salario',
                        color='departamento', size='edad',
                        title='Salario vs Experiencia')
        st.plotly_chart(fig, use_container_width=True)

def show_project_management(df_obras, df_asistencias, df_empleados):
    st.markdown('<div class="section-header">🏗️ Gestión de Obras</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("📋 Estado de Obras")
        
        for _, obra in df_obras.iterrows():
            with st.expander(f"🏗️ {obra['nombre']} - {obra['ubicacion']} | 💰 ${obra['presupuesto']:,.0f}"):
                col_a, col_b, col_c = st.columns(3)
                
                with col_a:
                    st.write(f"**Estado:** {obra['estado']}")
                    st.write(f"**Complejidad:** {obra['complejidad']}")
                    st.write(f"**Gerente:** {obra['gerente']}")
                
                with col_b:
                    # Calcular progreso basado en fecha
                    fecha_inicio = obra['fecha_inicio']
                    duracion = obra['duracion_estimada']
                    dias_transcurridos = (datetime.now() - fecha_inicio).days
                    progreso = min(95, max(5, (dias_transcurridos / duracion) * 100))
                    
                    st.write(f"**Progreso:** {progreso:.1f}%")
                    st.progress(progreso/100)
                
                with col_c:
                    # Empleados asignados a esta obra
                    empleados_obra = df_asistencias[df_asistencias['obra_id'] == obra['id']]['empleado_id'].nunique()
                    st.write(f"**Empleados:** {empleados_obra}")
                    
                    # Productividad en esta obra
                    productividad_obra = df_asistencias[df_asistencias['obra_id'] == obra['id']]['productividad'].mean()
                    st.write(f"**Productividad:** {productividad_obra:.1f}%")
    
    with col2:
        st.subheader("📈 Métricas de Obras")
        
        # KPIs de obras
        obras_en_progreso = len(df_obras[df_obras['estado'] == 'En Progreso'])
        obras_en_riesgo = len(df_obras[df_obras['estado'] == 'En Riesgo'])
        presupuesto_total = df_obras['presupuesto'].sum()
        
        st.metric("Obras en Progreso", obras_en_progreso)
        st.metric("Obras en Riesgo", obras_en_riesgo)
        st.metric("Presupuesto Total", f"${presupuesto_total:,.0f}")
        
        # Gráfico de estado de obras
        fig = px.pie(df_obras, names='estado', title='Distribución por Estado')
        st.plotly_chart(fig, use_container_width=True)
        
        # Gráfico de complejidad
        fig = px.histogram(df_obras, x='complejidad', title='Distribución por Complejidad')
        st.plotly_chart(fig, use_container_width=True)

def show_advanced_analytics(df_empleados, df_asistencias):
    st.markdown('<div class="section-header">📈 Analytics Avanzado</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🔍 Análisis de Correlaciones")
        
        # Preparar datos para correlación
        analytics_data = df_empleados.merge(
            df_asistencias.groupby('empleado_id').agg({
                'productividad': 'mean',
                'horas_extra': 'sum',
                'ausente': 'sum'
            }).reset_index(),
            left_on='id', right_on='empleado_id'
        )
        
        # Matriz de correlación
        corr_matrix = analytics_data[['salario', 'experiencia_meses', 'edad', 'productividad', 'horas_extra']].corr()
        
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, ax=ax)
        plt.title('Matriz de Correlación')
        st.pyplot(fig)
    
    with col2:
        st.subheader("📊 Segmentación de Personal")
        
        # Crear segmentos basados en salario y productividad
        analytics_data['segmento'] = pd.cut(
            analytics_data['salario'],
            bins=[0, 80000, 120000, float('inf')],
            labels=['Bajo', 'Medio', 'Alto']
        )
        
        fig = px.scatter(analytics_data, x='salario', y='productividad',
                        color='segmento', size='experiencia_meses',
                        hover_data=['nombre', 'departamento'],
                        title='Segmentación por Salario y Productividad')
        st.plotly_chart(fig, use_container_width=True)
    
    # Análisis predictivo simulado
    st.subheader("🎯 Modelo Predictivo - Riesgo de Rotación")
    
    col1, col2, col3 = st.columns(3)
    
    # Simular predicciones
    empleados_riesgo = df_empleados.sample(5)
    
    for _, emp in empleados_riesgo.iterrows():
        riesgo = np.random.uniform(0.6, 0.9)
        
        with col1:
            if riesgo > 0.8:
                st.markdown(f'<div class="alert-high">🚨 {emp["nombre"]} {emp["apellido"]}<br>Riesgo: {riesgo:.0%}</div>', unsafe_allow_html=True)
        with col2:
            if 0.7 <= riesgo <= 0.8:
                st.markdown(f'<div class="alert-medium">⚠️ {emp["nombre"]} {emp["apellido"]}<br>Riesgo: {riesgo:.0%}</div>', unsafe_allow_html=True)
        with col3:
            if riesgo < 0.7:
                st.info(f"✅ {emp['nombre']} {emp['apellido']}\nRiesgo: {riesgo:.0%}")

def show_performance_analytics(df_empleados, df_asistencias):
    st.markdown('<div class="section-header">🎯 Análisis de Desempeño</div>', unsafe_allow_html=True)
    
    # Combinar datos de empleados con asistencias
    performance_data = df_empleados.merge(
        df_asistencias.groupby('empleado_id').agg({
            'productividad': 'mean',
            'calidad_trabajo': 'mean',
            'horas_extra': 'sum',
            'ausente': 'sum'
        }).reset_index(),
        left_on='id', right_on='empleado_id'
    )
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🏆 Top Performers")
        
        # Top 10 por productividad
        top_performers = performance_data.nlargest(10, 'productividad')
        
        for _, emp in top_performers.iterrows():
            st.write(f"**{emp['nombre']} {emp['apellido']}** - {emp['departamento']}")
            st.write(f"Productividad: {emp['productividad']:.1f}% | Calidad: {emp['calidad_trabajo']:.1f}%")
            st.progress(emp['productividad']/100)
            st.markdown("---")
    
    with col2:
        st.subheader("📊 Distribución de Desempeño")
        
        # Histograma de productividad
        fig = px.histogram(performance_data, x='productividad', 
                          nbins=20, title='Distribución de Productividad')
        st.plotly_chart(fig, use_container_width=True)
        
        # Box plot por departamento
        fig = px.box(performance_data, x='departamento', y='productividad',
                    title='Productividad por Departamento')
        st.plotly_chart(fig, use_container_width=True)
    
    # Análisis de ausentismo
    st.subheader("🏥 Análisis de Ausentismo")
    
    col1, col2 = st.columns(2)
    
    with col1:
        ausentismo_depto = performance_data.groupby('departamento')['ausente'].mean().sort_values(ascending=False)
        fig = px.bar(x=ausentismo_depto.index, y=ausentismo_depto.values,
                    title='Ausentismo Promedio por Departamento')
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Relación entre ausentismo y productividad
        fig = px.scatter(performance_data, x='ausente', y='productividad',
                        color='departamento', trendline='ols',
                        title='Ausentismo vs Productividad')
        st.plotly_chart(fig, use_container_width=True)

def show_early_warnings(df_empleados, df_obras, df_asistencias):
    st.markdown('<div class="section-header">⚠️ Sistema de Alertas Tempranas</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🚨 Alertas Críticas")
        
        # Alertas de obras en riesgo
        obras_riesgo = df_obras[df_obras['estado'] == 'En Riesgo']
        for _, obra in obras_riesgo.iterrows():
            st.error(f"**{obra['nombre']}** - {obra['ubicacion']}\nPresupuesto: ${obra['presupuesto']:,.0f}")
        
        # Alertas de alta rotación
        st.warning("**Rotación Alta en Electricidad** - 15% vs promedio 8%")
        st.warning("**Baja Productividad en Pintura** - 72% vs promedio 85%")
    
    with col2:
        st.subheader("🔔 Alertas Preventivas")
        
        # Empleados con alto ausentismo
        ausentismo_data = df_asistencias.groupby('empleado_id')['ausente'].sum()
        alto_ausentismo = ausentismo_data[ausentismo_data > 5]
        
        if len(alto_ausentismo) > 0:
            for emp_id, dias in alto_ausentismo.items():
                emp = df_empleados[df_empleados['id'] == emp_id].iloc[0]
                st.warning(f"**{emp['nombre']} {emp['apellido']}** - {dias} días de ausencia")
        
        # Obras con bajo rendimiento
        rendimiento_obras = df_asistencias.groupby('obra_id')['productividad'].mean()
        bajo_rendimiento = rendimiento_obras[rendimiento_obras < 75]
        
        if len(bajo_rendimiento) > 0:
            for obra_id, prod in bajo_rendimiento.items():
                obra = df_obras[df_obras['id'] == obra_id].iloc[0]
                st.info(f"**{obra['nombre']}** - Productividad: {prod:.1f}%")
    
    # Panel de control de alertas
    st.subheader("⚙️ Configuración de Alertas")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        umbral_productividad = st.slider("Umbral Baja Productividad (%)", 50, 90, 75)
        umbral_ausentismo = st.slider("Umbral Alto Ausentismo (días/mes)", 1, 10, 5)
    
    with col2:
        umbral_rotacion = st.slider("Umbral Alta Rotación (%)", 5, 20, 12)
        umbral_horas_extra = st.slider("Umbral Horas Extra (h/semana)", 5, 20, 10)
    
    with col3:
        if st.button("💾 Guardar Configuración", use_container_width=True):
            st.success("Configuración de alertas guardada exitosamente!")

def show_configuration():
    st.markdown('<div class="section-header">⚙️ Configuración del Sistema</div>', unsafe_allow_html=True)
    
    tab1, tab2, tab3, tab4 = st.tabs(["👥 Departamentos", "🏷️ Cargos", "📊 Métricas", "🔐 Usuarios"])
    
    with tab1:
        st.subheader("Gestión de Departamentos")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Departamentos Existentes**")
            deptos = ['Albañilería', 'Electricidad', 'Plomería', 'Herrería', 'Pintura']
            for depto in deptos:
                st.write(f"• {depto}")
        
        with col2:
            st.write("**Agregar Nuevo Departamento**")
            nuevo_depto = st.text_input("Nombre del departamento")
            if st.button("➕ Agregar Departamento"):
                if nuevo_depto:
                    st.success(f"Departamento '{nuevo_depto}' agregado exitosamente!")
    
    with tab2:
        st.subheader("Gestión de Cargos y Niveles")
        
        # Configuración de niveles salariales
        st.write("**Escala Salarial por Departamento**")
        
        niveles_data = {
            'Departamento': ['Albañilería', 'Electricidad', 'Plomería', 'Herrería', 'Pintura'],
            'Junior': [60000, 70000, 65000, 80000, 55000],
            'Semi-Senior': [80000, 90000, 85000, 100000, 70000],
            'Senior': [100000, 110000, 95000, 120000, 85000]
        }
        
        df_niveles = pd.DataFrame(niveles_data)
        st.dataframe(df_niveles, use_container_width=True)
    
    with tab3:
        st.subheader("Configuración de KPIs")
        
        kpis = [
            {"nombre": "Productividad", "activo": True, "meta": 85, "peso": 30},
            {"nombre": "Calidad", "activo": True, "meta": 90, "peso": 25},
            {"nombre": "Ausentismo", "activo": True, "meta": 3, "peso": 20},
            {"nombre": "Horas Extra", "activo": False, "meta": 10, "peso": 15},
            {"nombre": "Rotación", "activo": True, "meta": 8, "peso": 10}
        ]
        
        for kpi in kpis:
            col1, col2, col3, col4 = st.columns([2, 1, 1, 1])
            with col1:
                st.write(f"**{kpi['nombre']}**")
            with col2:
                st.checkbox("Activo", value=kpi['activo'], key=f"act_{kpi['nombre']}")
            with col3:
                st.number_input("Meta", value=kpi['meta'], key=f"meta_{kpi['nombre']}")
            with col4:
                st.number_input("Peso %", value=kpi['peso'], key=f"peso_{kpi['nombre']}")
            st.markdown("---")
    
    with tab4:
        st.subheader("Gestión de Usuarios")
        
        usuarios = [
            {"usuario": "admin", "rol": "Administrador", "email": "admin@empresa.com", "activo": True},
            {"usuario": "gerente.rrhh", "rol": "Gerente RRHH", "email": "rrhh@empresa.com", "activo": True},
            {"usuario": "supervisor.obra", "rol": "Supervisor", "email": "supervisor@empresa.com", "activo": True}
        ]
        
        for usuario in usuarios:
            col1, col2, col3, col4 = st.columns([2, 2, 2, 1])
            with col1:
                st.write(f"**{usuario['usuario']}**")
            with col2:
                st.write(usuario['rol'])
            with col3:
                st.write(usuario['email'])
            with col4:
                st.checkbox("Activo", value=usuario['activo'], key=f"user_{usuario['usuario']}")
            st.markdown("---")

if __name__ == "__main__":
    main()
