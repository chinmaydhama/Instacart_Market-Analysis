import pandas as pd
import dash
from dash.dependencies import Input, Output
from dash import dcc, html
from dash.exceptions import PreventUpdate
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
from dash import html
import random
from scipy.stats import probplot, shapiro, kstest, anderson, normaltest
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import scipy.stats as stats
import numpy as np
import zipfile
import os

if os.path.exists('final_data_instacart_400k.zip'):
    with zipfile.ZipFile('final_data_instacart_400k.zip', 'r') as zip_ref:
        zip_ref.extractall()
df_sliced = pd.read_csv('final_data_instacart_400k.csv')
df=df_sliced.copy()
data=df.copy()
necessary_columns = ['department_id', 'order_id', 'product_id', 'product_name', 'reordered', 'order_hour_of_day', 'add_to_cart_order', 'aisle_id']
for column in necessary_columns:
    if column not in df_sliced.columns:
        print(f"Error: Column '{column}' not found in the DataFrame.")
        exit(1)
department_dict = df_sliced[['department_id', 'department']].set_index('department_id')['department'].to_dict()
dropdown_options = [{'label': i, 'value': i} for i in df.columns]
imputer = SimpleImputer(strategy='mean')
numeric_cols = ['add_to_cart_order', 'order_dow', 'order_hour_of_day', 'days_since_prior_order']
df[numeric_cols] = imputer.fit_transform(df[numeric_cols])

# Standardize the numeric columns for PCA
df_standardized = StandardScaler().fit_transform(df[numeric_cols])
# Replace the department IDs with the corresponding department names
df_sliced['department_id'] = df_sliced['department_id'].map(department_dict)

# Get the unique aisle names for the dropdown
aisle_names = df_sliced['aisle_id'].unique()
total_customers = df['order_id'].nunique()
total_products = df['product_id'].nunique()
avg_order_per_customer = df['product_id'].count() / total_customers
avg_order_per_customer = round(avg_order_per_customer, 2)
Total_orders = df['order_id'].count()
dummy_values = [total_customers, total_products, avg_order_per_customer, Total_orders]

# Instacart brand colors
INSTACART_GREEN = "#0AAD05"
INSTACART_ORANGE = "#FF7009"
INSTACART_DARK_GREEN = "#003D29"
INSTACART_GREEN_LIGHT = "#7ED957"
INSTACART_ORANGE_LIGHT = "#FFB380"

# Plotly theme: Instacart-aligned, clean
PLOT_TEMPLATE = dict(
    layout=dict(
        font=dict(family="DM Sans, system-ui, sans-serif", size=12, color="#0f172a"),
        title=dict(font=dict(size=16, color="#0f172a"), x=0.02, xanchor="left"),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="#f8fafc",
        margin=dict(t=56, b=48, l=56, r=32),
        hoverlabel=dict(bgcolor="#fff", font_size=12, font_family="DM Sans"),
        xaxis=dict(showgrid=True, gridcolor="#e2e8f0", zeroline=False, tickfont=dict(size=11)),
        yaxis=dict(showgrid=True, gridcolor="#e2e8f0", zeroline=False, tickfont=dict(size=11)),
        colorway=[INSTACART_GREEN, INSTACART_ORANGE, INSTACART_DARK_GREEN, INSTACART_GREEN_LIGHT, INSTACART_ORANGE_LIGHT, "#64748b"],
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, font=dict(size=11)),
    )
)

def apply_theme(fig, height=None):
    """Apply production-grade theme to a Plotly figure."""
    fig.update_layout(**PLOT_TEMPLATE["layout"])
    if height:
        fig.update_layout(height=height)
    try:
        fig.update_traces(marker=dict(line=dict(width=0)), selector=dict(type="bar"))
    except Exception:
        pass
    return fig

icons = ["fa-users", "fa-box-open", "fa-shopping-cart", "fa-chart-line"]
descriptions = ["Total Customers", "Total Products", "Avg Order per Customer", "Total Orders"]

# Single Dash app with external stylesheets (assets/styles.css is auto-loaded from assets/)
app = dash.Dash(__name__, suppress_callback_exceptions=True, external_stylesheets=[
    "https://use.fontawesome.com/releases/v5.8.1/css/all.css",
])

app.layout = html.Div(id="main-container", children=[
    html.Header(className="dash-header", children=[
        html.H1("Instacart Market Analysis"),
        html.P("Grocery basket analytics, department & aisle performance, and statistical insights"),
    ]),

    html.Div(className="kpi-grid", children=[
        html.Div(className="kpi-card", children=[
            html.I(className=f"fas {icon} kpi-icon"),
            html.Div(str(value), className="kpi-value"),
            html.Div(description, className="kpi-label"),
        ]) for icon, value, description in zip(icons, dummy_values, descriptions)
    ]),

    html.Div(className="Tab-container", children=[
        dcc.Tabs(id="tabs", value="tab-1", children=[
            dcc.Tab(label="Department Analysis", value="tab-1"),
            dcc.Tab(label="Aisle Analysis", value="tab-2"),
            dcc.Tab(label="Product Insights", value="tab-3"),
            dcc.Tab(label="Statistical Analysis", value="tab-4"),
        ]),
        html.Div(id="tabs-content", className="tab-content-inner"),
    ]),
])


@app.callback(Output('tabs-content', 'children'),
              Input('tabs', 'value'))
def render_content(tab):
    if tab == 'tab-1':
        return html.Div([
            html.Div(className="control-panel", children=[
                html.Div([
                    html.Label("Department"),
                    dcc.Dropdown(
                        id='department-dropdown',
                        options=[{'label': i, 'value': i} for i in df_sliced['department_id'].unique()],
                        value=df_sliced['department_id'].unique()[0],
                        clearable=False,
                    ),
                ]),
                html.Div([
                    html.Label("Top N products"),
                    dcc.RadioItems(
                        id='dept-top-n-radio',
                        options=[{'label': 'Top 5', 'value': 5}, {'label': 'Top 10', 'value': 10}],
                        value=10,
                        inline=True,
                    ),
                ]),
            ]),
            html.Div(className="metric-strip", children=[
                html.Div(id='total-orders', className="metric-box"),
                html.Div(id='total-products', className="metric-box"),
                html.Div(id='total-aisles', className="metric-box"),
            ]),
            html.Div(className="chart-card", children=[dcc.Graph(id='subplots-graph', config={'displayModeBar': True, 'displaylogo': False})]),
        ])
    elif tab == 'tab-2':
        return html.Div([
            html.Div(className="control-panel", children=[
                html.Div([
                    html.Label("Aisle"),
                    dcc.Dropdown(
                        id='aisle-dropdown',
                        options=[{'label': str(i), 'value': i} for i in aisle_names],
                        value=aisle_names[0],
                        clearable=False,
                    ),
                ]),
                html.Div([
                    html.Label("Show"),
                    dcc.RadioItems(
                        id='aisle-top-n-radio',
                        options=[{'label': i, 'value': i} for i in ['Top 5', 'Top 10', 'All']],
                        value='Top 5',
                        inline=True,
                    ),
                ]),
            ]),
            html.Div(className="chart-card", children=[dcc.Graph(id='graph', config={'displayModeBar': True, 'displaylogo': False})]),
        ])
    elif tab == 'tab-3':
        return html.Div([
            html.H2("Product Insights", className="section-title"),
            html.Div(className="control-panel", children=[
                html.Div([
                    html.Label("Product"),
                    dcc.Dropdown(
                        id='product-selector',
                        options=[{'label': product, 'value': product} for product in df['product_name'].unique()],
                        value=df['product_name'].unique()[0],
                        clearable=False,
                    ),
                ]),
            ]),
            html.Div(className="chart-row", children=[
                html.Div(className="chart-card", children=[dcc.Graph(id='gauge-chart', config={'displayModeBar': True, 'displaylogo': False})]),
                html.Div(className="chart-card", children=[dcc.Graph(id='pie-chart', config={'displayModeBar': True, 'displaylogo': False})]),
            ]),
            html.Div(className="chart-row", children=[
                html.Div(className="chart-card", children=[dcc.Graph(id='sales-volume-chart', config={'displayModeBar': True, 'displaylogo': False})]),
                html.Div(className="chart-card", children=[dcc.Graph(id='cart-position-chart', config={'displayModeBar': True, 'displaylogo': False})]),
            ]),
        ])
    elif tab== 'tab-4':
        return html.Div([
            html.H2("Statistical Analysis", className="section-title"),
            html.Div(className="chart-row", children=[
                html.Div(className="chart-card", style={"flex": "1 1 33%"}, children=[
                    html.Label("Feature (boxplot)", style={"display": "block", "marginBottom": "8px"}),
                    dcc.Dropdown(id='feature-dropdown', options=dropdown_options, value='add_to_cart_order', clearable=False),
                    dcc.Graph(id='combined-boxplot', config={'displayModeBar': True, 'displaylogo': False}),
                ]),
                html.Div(className="chart-card", style={"flex": "1 1 33%"}, children=[
                    html.Label("Normality: column", style={"display": "block", "marginBottom": "8px"}),
                    dcc.Dropdown(id='normality-column', options=dropdown_options, value='days_since_prior_order', clearable=False),
                    dcc.Dropdown(id='normality-test', options=[
                        {'label': 'Shapiro-Wilk', 'value': 'shapiro'}, {'label': 'Kolmogorov-Smirnov', 'value': 'ks'},
                        {'label': "Anderson-Darling", 'value': 'anderson'}, {'label': "D'Agostino's K-squared", 'value': 'dagostino'}
                    ], value='shapiro', clearable=False),
                    dcc.Graph(id='qq-plot', config={'displayModeBar': True, 'displaylogo': False}),
                    dcc.Slider(id='slider', min=1, max=5000, step=500, value=1000, marks={i: str(i) for i in range(0, 5001, 500)}),
                    html.Div(id='normality-test-result', style={'fontSize': '14px', 'marginTop': '8px', 'color': '#64748b'}),
                ]),
                html.Div(className="chart-card", style={"flex": "1 1 33%"}, children=[
                    html.Label("Transformation", style={"display": "block", "marginBottom": "8px"}),
                    dcc.Dropdown(id='transformation-feature', options=dropdown_options, value='days_since_prior_order', clearable=False),
                    dcc.RadioItems(id='transformation-type', options=[
                        {'label': 'Log', 'value': 'log'}, {'label': 'Square root', 'value': 'sqrt'}
                    ], value='log', inline=True),
                    dcc.Graph(id='transformed-data', config={'displayModeBar': True, 'displaylogo': False}),
                ]),
            ]),
            html.Div(className="chart-row", children=[
                html.Div(className="chart-card", style={"flex": "1 1 50%"}, children=[
                    html.Label("Scatter: X / Y", style={"display": "block", "marginBottom": "8px"}),
                    dcc.Dropdown(id='xaxis-column', options=dropdown_options, value='order_number', clearable=False),
                    dcc.Dropdown(id='yaxis-column', options=dropdown_options, value='days_since_prior_order', clearable=False),
                    dcc.Checklist(id='trend-line', options=[{'label': 'Show trend line', 'value': 'show'}], value=[]),
                    dcc.Graph(id='interactive-scatter-plot', config={'displayModeBar': True, 'displaylogo': False}),
                    html.Div(id='r-squared-value', style={'fontSize': '14px', 'marginTop': '8px', 'color': '#64748b'}),
                ]),
                html.Div(className="chart-card", style={"flex": "1 1 50%"}, children=[
                    html.Label("PCA", style={"display": "block", "marginBottom": "8px"}),
                    dcc.Dropdown(id='pca-components', options=[{'label': f'{i} components', 'value': i} for i in range(1, len(numeric_cols) + 1)], value=2, clearable=False),
                    html.Button('Perform PCA', id='perform-pca-btn', n_clicks=0, className="dash-button"),
                    dcc.Graph(id='pca-analysis', config={'displayModeBar': True, 'displaylogo': False}),
                ]),
            ]),
            html.Div(className="chart-card", children=[
                dcc.Graph(id='correlation-heatmap', config={'displayModeBar': True, 'displaylogo': False}),
            ]),
        ])

# Product Insights: run when tab is Product Insights OR when product-selector changes
@app.callback(
    [Output('gauge-chart', 'figure'),
     Output('pie-chart', 'figure'),
     Output('sales-volume-chart', 'figure'),
     Output('cart-position-chart', 'figure')],
    [Input('tabs', 'value'), Input('product-selector', 'value')]
)
def update_graphs(tab_value, selected_product):
    if tab_value != 'tab-3':
        raise PreventUpdate
    product_options = df['product_name'].unique()
    if selected_product is None or selected_product not in product_options:
        selected_product = product_options[0]
    filtered_df = df[df['product_name'] == selected_product]
    if filtered_df.empty:
        selected_product = product_options[0]
        filtered_df = df[df['product_name'] == selected_product]
    dow_labels = ['Sun', 'Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat']

    reorder_rate = filtered_df['reordered'].mean()
    gauge_fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=round(reorder_rate, 3),
        number=dict(suffix="", font=dict(size=28)),
        domain={'x': [0.1, 0.9], 'y': [0.15, 0.85]},
        title={'text': "Reorder Rate", 'font': {'size': 14}},
        gauge={
            'axis': {'range': [0, 1], 'tickwidth': 1},
            'bar': {'color': INSTACART_GREEN},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "#e2e8f0",
            'steps': [{'range': [0, 0.33], 'color': "#f1f5f9"}, {'range': [0.33, 0.66], 'color': "#e2e8f0"}, {'range': [0.66, 1], 'color': "#cbd5e1"}],
            'threshold': {'line': {'color': INSTACART_GREEN, 'width': 4}, 'thickness': 0.8, 'value': reorder_rate},
        }
    ))
    apply_theme(gauge_fig, height=280)

    sales_dow = filtered_df.groupby('order_dow').size().reindex(range(7), fill_value=0)
    pie_fig = go.Figure(data=[go.Pie(
        labels=[dow_labels[i] for i in sales_dow.index],
        values=sales_dow.values,
        hole=0.5,
        marker=dict(colors=[INSTACART_GREEN, INSTACART_ORANGE, INSTACART_DARK_GREEN, INSTACART_GREEN_LIGHT, INSTACART_ORANGE_LIGHT, "#cbd5e1", "#e2e8f0"]),
        textinfo="percent",
        textposition="inside",
        insidetextorientation="horizontal",
        hovertemplate="%{label}<br>Orders: %{value}<extra></extra>",
    )])
    pie_fig.update_layout(title_text="Sales by Day of Week", uniformtext_minsize=10, uniformtext_mode="hide", showlegend=True, legend=dict(orientation="h", yanchor="top", y=-0.05))
    apply_theme(pie_fig, height=280)

    sales_volume = filtered_df.groupby('order_hour_of_day').size()
    sales_volume_fig = go.Figure()
    sales_volume_fig.add_trace(go.Scatter(
        x=sales_volume.index, y=sales_volume.values, fill='tozeroy',
        line=dict(color=INSTACART_GREEN, width=2), fillcolor="rgba(10, 173, 5, 0.15)",
        name='Orders',
    ))
    sales_volume_fig.update_layout(title_text="Orders by Hour of Day", xaxis_title="Hour", yaxis_title="Orders")
    apply_theme(sales_volume_fig, height=280)

    avg_cart_position = filtered_df['add_to_cart_order'].mean()
    cart_position_fig = go.Figure(go.Indicator(
        mode="number",
        value=round(avg_cart_position, 1),
        title={'text': "Avg Add-to-Cart Position", 'font': {'size': 14}},
        number=dict(font=dict(size=36, color=INSTACART_GREEN)),
    ))
    cart_position_fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", margin=dict(t=60, b=40))
    apply_theme(cart_position_fig, height=280)

    return gauge_fig, pie_fig, sales_volume_fig, cart_position_fig

# Callbacks for updating the plot
@app.callback(
    Output('subplots-graph', 'figure'),
    [Input('department-dropdown', 'value'), Input('dept-top-n-radio', 'value')]
)
def update_figures(selected_department, top_n):
    filtered_df = df_sliced[df_sliced['department_id'] == selected_department]
    top_n = 10 if top_n is None else top_n
    top_products = filtered_df['product_name'].value_counts().nlargest(top_n).index
    top_products_df = filtered_df[filtered_df['product_name'].isin(top_products)]

    fig = make_subplots(
        rows=2, cols=2,
        specs=[[{'type': 'pie'}, {'type': 'bar'}], [{'type': 'scatter'}, {'type': 'bar'}]],
        subplot_titles=("Top products (share)", "Reorder rate by product", "Orders by hour", "Orders by day of week"),
        vertical_spacing=0.14, horizontal_spacing=0.10,
    )

    top_product_counts = filtered_df['product_name'].value_counts().head(top_n)
    # Show only percent inside; keep labels in hover to reduce clutter
    fig.add_trace(
        go.Pie(labels=top_product_counts.index, values=top_product_counts.values, hole=0.5,
               marker=dict(colors=[INSTACART_GREEN, INSTACART_ORANGE, INSTACART_DARK_GREEN, INSTACART_GREEN_LIGHT, INSTACART_ORANGE_LIGHT, "#94a3b8", "#cbd5e1", "#e2e8f0", "#f1f5f9", "#f8fafc"][:top_n]),
               textinfo="percent", textposition="inside", insidetextorientation="horizontal",
               hovertemplate="%{label}<br>%{percent}<extra></extra>", showlegend=False),
        row=1, col=1,
    )
    reorder_rate = top_products_df.groupby('product_name')['reordered'].mean().sort_values(ascending=True)
    fig.add_trace(
        go.Bar(y=reorder_rate.index, x=reorder_rate.values, orientation='h', marker_color=INSTACART_GREEN, showlegend=False),
        row=1, col=2,
    )
    order_hour_trend = filtered_df.groupby('order_hour_of_day').size()
    fig.add_trace(
        go.Scatter(x=order_hour_trend.index, y=order_hour_trend.values, mode='lines+markers',
                   line=dict(color=INSTACART_GREEN, width=2), marker=dict(size=6), showlegend=False),
        row=2, col=1,
    )
    order_day_trend = filtered_df.groupby('order_dow').size()
    dow_labels = ['Sun', 'Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat']
    fig.add_trace(
        go.Bar(x=[dow_labels[i] for i in order_day_trend.index], y=order_day_trend.values, marker_color=INSTACART_ORANGE, showlegend=False),
        row=2, col=2,
    )

    fig.update_xaxes(title_text="Hour", row=2, col=1)
    fig.update_yaxes(title_text="Orders", row=2, col=1)
    fig.update_xaxes(title_text="Day", row=2, col=2)
    fig.update_yaxes(title_text="Orders", row=2, col=2)
    fig.update_xaxes(title_text="Reorder rate", row=1, col=2)
    fig.update_yaxes(title_text="", row=1, col=2)
    apply_theme(fig, height=700)
    return fig

# Callbacks for updating the numbers
@app.callback(
    Output('total-orders', 'children'),
    Output('total-products', 'children'),
    Output('total-aisles', 'children'),
    Input('department-dropdown', 'value')
)
def update_numbers(selected_department):
    filtered_df = df_sliced[df_sliced['department_id'] == selected_department]
    total_orders = filtered_df['order_id'].nunique()
    total_products = filtered_df['product_id'].nunique()
    total_aisles = filtered_df['aisle_id'].nunique()
    return f'Total Orders: {total_orders}', f'Total Products: {total_products}', f'Total Aisles: {total_aisles}'

# Callbacks for updating the aisle analysis plot
@app.callback(
    Output('graph', 'figure'),
    Input('aisle-dropdown', 'value'),
    Input('aisle-top-n-radio', 'value')
)
def update_aisle_graph(selected_aisle, top_n):
    aisle_data = df_sliced[df_sliced['aisle_id'] == selected_aisle]
    top_n = top_n or 'Top 5'
    n = 5 if top_n == 'Top 5' else (10 if top_n == 'Top 10' else len(aisle_data))

    order_counts = aisle_data['product_name'].value_counts().head(n)
    top_reorders = aisle_data[aisle_data['reordered'] == 1]['product_name'].value_counts().head(n)
    cart_order = aisle_data['add_to_cart_order']
    department_sales = aisle_data.groupby('department_id')['order_id'].count()

    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=("Order frequency (top products)", "Top reordered products", "Add-to-cart position", "Department share"),
        specs=[[{'type': 'bar'}, {'type': 'bar'}], [{'type': 'histogram'}, {'type': 'domain'}]],
        vertical_spacing=0.14, horizontal_spacing=0.10,
    )
    fig.add_trace(
        go.Bar(x=order_counts.values, y=order_counts.index, orientation='h', marker_color=INSTACART_GREEN, showlegend=False),
        row=1, col=1,
    )
    fig.add_trace(
        go.Bar(x=top_reorders.values, y=top_reorders.index, orientation='h', marker_color=INSTACART_ORANGE, showlegend=False),
        row=1, col=2,
    )
    fig.add_trace(
        go.Histogram(x=cart_order, nbinsx=min(30, max(10, cart_order.nunique())), marker_color=INSTACART_GREEN, showlegend=False),
        row=2, col=1,
    )
    fig.add_trace(
        go.Pie(labels=department_sales.index, values=department_sales.values, hole=0.5,
               marker=dict(colors=[INSTACART_GREEN, INSTACART_ORANGE, INSTACART_DARK_GREEN, INSTACART_GREEN_LIGHT, INSTACART_ORANGE_LIGHT]), showlegend=False),
        row=2, col=2,
    )
    fig.update_xaxes(title_text="Orders", row=1, col=1)
    fig.update_xaxes(title_text="Reorders", row=1, col=2)
    fig.update_xaxes(title_text="Cart position", row=2, col=1)
    fig.update_yaxes(title_text="Count", row=2, col=1)
    apply_theme(fig, height=800)
    return fig

# Callbacks to update plots based on dropdown selections and button clicks
@app.callback(
    Output('combined-boxplot', 'figure'),
    Input('feature-dropdown', 'value')
)
def update_plots(selected_feature):
    Q1 = df[selected_feature].quantile(0.25)
    Q3 = df[selected_feature].quantile(0.75)
    IQR = Q3 - Q1
    df_no_outliers = df[(df[selected_feature] >= (Q1 - 1.5 * IQR)) & (df[selected_feature] <= (Q3 + 1.5 * IQR))]

    combined_fig = make_subplots(rows=1, cols=2, subplot_titles=("With outliers", "Without outliers (IQR filter)"))
    combined_fig.add_trace(go.Box(y=df[selected_feature], name="With outliers", marker_color=INSTACART_GREEN, line_color=INSTACART_DARK_GREEN), row=1, col=1)
    combined_fig.add_trace(go.Box(y=df_no_outliers[selected_feature], name="Filtered", marker_color=INSTACART_ORANGE, line_color=INSTACART_DARK_GREEN), row=1, col=2)
    combined_fig.update_layout(showlegend=False)
    apply_theme(combined_fig, height=360)
    return combined_fig

@app.callback(
    Output('pca-analysis', 'figure'),
    [Input('perform-pca-btn', 'n_clicks'),
     Input('pca-components', 'value')],
    prevent_initial_call=True
)
def update_pca(n_clicks, n_components):
    if n_clicks > 0:
        pca = PCA(n_components=n_components)
        pca.fit_transform(df_standardized)
        explained_var = pca.explained_variance_ratio_
        fig = go.Figure(go.Bar(x=[f'PC{i+1}' for i in range(n_components)], y=explained_var, marker_color=INSTACART_GREEN))
        fig.update_layout(title_text=f"Explained variance ({n_components} components)", xaxis_title="Component", yaxis_title="Variance explained", yaxis_tickformat=".0%")
        apply_theme(fig, height=360)
        return fig
    return go.Figure().add_annotation(text="Click 'Perform PCA' to run", x=0.5, y=0.5, showarrow=False, font=dict(size=14))

@app.callback(
    [Output('qq-plot', 'figure'),
     Output('normality-test-result', 'children')],
    [Input('normality-column', 'value'),
     Input('normality-test', 'value'),
     Input('slider', 'value')]
)
def update_qq_and_test(selected_feature, test_type, n_value):
    sample = df[selected_feature].dropna()
    n_value = min(n_value, len(sample))
    data = sample.sample(n=n_value, replace=False, random_state=1)
    if test_type == 'shapiro':
        stat, p = shapiro(data)
        test_name = "Shapiro-Wilk"
    elif test_type == 'ks':
        stat, p = kstest(data, 'norm')
        test_name = "Kolmogorov-Smirnov"
    elif test_type == 'anderson':
        result = anderson(data)
        stat = result.statistic
        p = result.critical_values[2]  # approximate
        test_name = "Anderson-Darling"
    else:
        stat, p = normaltest(data)
        test_name = "D'Agostino K²"

    (osm, osr), (slope, intercept, r) = stats.probplot(data, dist="norm")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=osm, y=osr * slope + intercept, mode='lines', name='Theoretical', line=dict(color="#94a3b8", dash="dash")))
    fig.add_trace(go.Scatter(x=osm, y=osr, mode='markers', name='Data', marker=dict(color=INSTACART_GREEN, size=6, line=dict(width=0))))
    fig.update_layout(title_text=f"Q-Q plot: {selected_feature}", xaxis_title="Theoretical quantiles", yaxis_title="Sample quantiles")
    apply_theme(fig, height=320)
    result_text = f"{test_name}: stat = {stat:.3f}, p = {p:.4f}"
    return fig, result_text

@app.callback(
    Output('transformed-data', 'figure'),
    [Input('transformation-feature', 'value'),
     Input('transformation-type', 'value')]
)
def update_transformed_data(selected_feature, transformation):
    raw = df[selected_feature].dropna()
    if transformation == 'log':
        transformed = np.log1p(raw)
        title = f"Log(1 + {selected_feature})"
    else:
        transformed = np.sqrt(raw)
        title = f"√{selected_feature}"
    fig = go.Figure(go.Histogram(x=transformed, nbinsx=min(40, max(15, int(transformed.nunique() / 2))), marker_color=INSTACART_GREEN))
    fig.update_layout(title_text=title, xaxis_title="Value", yaxis_title="Count")
    apply_theme(fig, height=320)
    return fig

# Precompute the correlation matrix
corr_matrix = df[numeric_cols].corr()

@app.callback(
    Output('correlation-heatmap', 'figure'),
    Input('transformation-type', 'value')
)
def update_heatmap(_):
    fig = go.Figure(go.Heatmap(
        z=corr_matrix.values, x=corr_matrix.columns, y=corr_matrix.index,
        colorscale=[[0, "#f0fdf4"], [0.5, INSTACART_GREEN_LIGHT], [1, INSTACART_GREEN]],
        text=np.round(corr_matrix.values, 2), texttemplate="%{text}", textfont=dict(size=11),
    ))
    fig.update_layout(title_text="Correlation matrix", xaxis_title="", yaxis_title="")
    apply_theme(fig, height=400)
    return fig


@app.callback(
    Output('interactive-scatter-plot', 'figure'),
    Output('r-squared-value', 'children'),
    Input('xaxis-column', 'value'),
    Input('yaxis-column', 'value'),
    Input('trend-line', 'value')
)
def update_scatter(xaxis_column_name, yaxis_column_name, trend_line_value):
    show_trend = 'show' in (trend_line_value or [])
    fig = px.scatter(
        data_frame=data.sample(n=min(3000, len(data)), random_state=42),
        x=xaxis_column_name,
        y=yaxis_column_name,
        opacity=0.6,
        trendline="ols" if show_trend else None,
    )
    fig.update_traces(marker=dict(size=6, line=dict(width=0)), selector=dict(mode='markers'))
    fig.update_layout(title_text=f"{xaxis_column_name} vs {yaxis_column_name}", xaxis_title=xaxis_column_name, yaxis_title=yaxis_column_name)
    r_squared = ""
    if show_trend:
        try:
            results = px.get_trendline_results(fig)
            r2 = results.px_fit_results.iloc[0].rsquared
            r_squared = f"R² = {r2:.4f}"
            fig.add_annotation(x=0.02, y=0.98, text=r_squared, xref="paper", yref="paper", showarrow=False, font=dict(size=12), bgcolor="rgba(255,255,255,0.8)")
        except Exception:
            pass
    apply_theme(fig, height=360)
    return fig, r_squared

# Expose Flask server for Gunicorn (production deploy on Render/Railway)
server = app.server

if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0', port=8050)