import pandas as pd
import dash
from dash.dependencies import Input, Output
from dash import dcc, html
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


# Load the data
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

# Create a Dash application
app = dash.Dash(__name__,suppress_callback_exceptions=True)

# Get the unique aisle names for the dropdown
aisle_names = df_sliced['aisle_id'].unique()
total_customers = df['order_id'].nunique()
total_products = df['product_id'].nunique()
avg_order_per_customer = df['product_id'].count() / total_customers
avg_order_per_customer = round(avg_order_per_customer, 2)
Total_orders = df['order_id'].count()

# Without a specific definition of "churn" or a time frame to consider, it's not possible to calculate churn %
# For the purpose of this example, let's assume that churn % is 0
churn_percentage = 0

# Align the calculated metrics with dummy_values
dummy_values = [total_customers, total_products, avg_order_per_customer, Total_orders]



# Initialize the Dash app with external stylesheets
external_stylesheets = ['https://use.fontawesome.com/releases/v5.8.1/css/all.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

# Define the icons you want to use (Font Awesome classes)
icons = [
    "fa-users",  # For Total Customers
    "fa-box-open",  # For Total Products
    "fa-shopping-cart",  # For Avg Order per customer
    "fa-chart-line"  # For Churn %
]

# Text descriptions for each metric
descriptions = [
    "Total Customers",
    "Total Products",
    "Avg Order per customer",
    "Total Orders"
]

app.layout = html.Div(children=[
    html.H1(children='Instacart-Grocery-Market-Analysis', style={'textAlign': 'center'}),

    # Display dummy values in styled boxes with icons
    html.Div(children=[
        html.Div(children=[
            html.I(className=f"fas {icon}", style={'fontSize': '24px', 'color': '#FF8C00'}),
            html.P(str(value), style={'fontSize': '28px', 'fontWeight': 'bold'}),
            html.P(description, style={'fontSize': '15px'})
        ], style={
            'border': '1px solid #ddd',
            'borderLeft': '5px solid #FF8C00',
            'borderRadius': '5px',
            'padding': '5px 10px',
            'margin': '10px',
            'textAlign': 'center',
            'display': 'inline-block',
            'minWidth': '200px'
        }) for icon, value, description in zip(icons, dummy_values, descriptions)
    ], style={'textAlign': 'center', 'display': 'flex', 'justifyContent': 'space-around', 'flexWrap': 'wrap'}),

    dcc.Tabs(id='tabs', value='tab-1', children=[
        dcc.Tab(label='Department Analysis', value='tab-1'),
        dcc.Tab(label='Aisle Analysis', value='tab-2'),
        dcc.Tab(label='Product Insights Dashboard', value='tab-3'),
        dcc.Tab(label='Statistical Analysis', value='tab-4')
    ]),
    html.Div(id='tabs-content')
])


@app.callback(Output('tabs-content', 'children'),
              Input('tabs', 'value'))
def render_content(tab):
    if tab == 'tab-1':
        return html.Div([
            dcc.Dropdown(
                id='department-dropdown',
                options=[{'label': i, 'value': i} for i in df_sliced['department_id'].unique()],
                value=df_sliced['department_id'].unique()[0]  # default value
            ),
            dcc.RadioItems(
                id='top-n-radio',
                options=[{'label': 'Top 5', 'value': 5}, {'label': 'Top 10', 'value': 10}],
                value=10  # default value
            ),
            html.Div([
                html.Div(id='total-orders', style={
                    'textAlign': 'center',
                    'color': '#28282B',
                    'fontSize': 20,
                    'padding': '10px',
                    'border': '2px solid #28282B'
                }),
                html.Div(id='total-products', style={
                    'textAlign': 'center',
                    'color': '#28282B',
                    'fontSize': 20,
                    'padding': '10px',
                    'border': '2px solid #28282B'
                }),
                html.Div(id='total-aisles', style={
                    'textAlign': 'center',
                    'color': '#28282B',
                    'fontSize': 20,
                    'padding': '10px',
                    'border': '2px solid #28282B'
                })
            ], style={'display': 'flex', 'justifyContent': 'space-around'}),

            dcc.Graph(id='subplots-graph')
        ])
    elif tab == 'tab-2':
        return html.Div([
            dcc.Dropdown(
                id='aisle-dropdown',
                options=[{'label': i, 'value': i} for i in aisle_names],
                value=aisle_names[0]
            ),
            dcc.RadioItems(
                id='top-n-radio',
                options=[{'label': i, 'value': i} for i in ['Top 5', 'Top 10', 'All']],
                value='Top 5'
            ),
            dcc.Graph(id='graph')
        ])
    elif tab == 'tab-3':
        return html.Div([
            html.H1("Product Insights Dashboard"),

            dcc.Dropdown(
                id='product-selector',
                options=[{'label': product, 'value': product} for product in df['product_name'].unique()],
                value=df['product_name'].unique()[0],  # Default selection
            ),

            html.Div([
                dcc.Graph(id='gauge-chart', style={"display": "inline-block", "width": "50%"}),
                dcc.Graph(id='pie-chart', style={"display": "inline-block", "width": "50%"})
            ]),

            html.Div([
                dcc.Graph(id='sales-volume-chart', style={"display": "inline-block", "width": "50%"}),
                dcc.Graph(id='cart-position-chart', style={"display": "inline-block", "width": "50%"})
            ]),
        ])
    elif tab== 'tab-4':
        return html.Div([
            html.Div([
                html.Div([
                    dcc.Dropdown(
                        id='feature-dropdown',
                        options=dropdown_options,
                        value='add_to_cart_order'
                    ),
                    dcc.Graph(id='combined-boxplot'),
                ], style={'width': '33%', 'display': 'inline-block', 'border': 'thin lightgrey solid',
                          'padding': '5px'}),

                html.Div([
                    dcc.Dropdown(
                        id='normality-column',
                        options=dropdown_options,
                        value='days_since_prior_order'
                    ),
                    dcc.Dropdown(
                        id='normality-test',
                        options=[
                            {'label': 'Shapiro-Wilk', 'value': 'shapiro'},
                            {'label': 'Kolmogorov-Smirnov', 'value': 'ks'},
                            {'label': "Anderson-Darling", 'value': 'anderson'},
                            {'label': "D'Agostino's K-squared", 'value': 'dagostino'}
                        ],
                        value='shapiro'
                    ),

                    dcc.Graph(id='qq-plot'),
                    dcc.Slider(
                        id='slider',
                        min=1,
                        max=5000,
                        step=500,
                        value=1000,
                        marks={i: str(i) for i in range(0, 5001, 500)}
                    ),
                    html.Div(id='normality-test-result', style={'font-size': '20px'}),
                ], style={'width': '33%', 'display': 'inline-block', 'border': 'thin lightgrey solid',
                          'padding': '5px'}),

                html.Div([
                    dcc.Dropdown(
                        id='transformation-feature',
                        options=dropdown_options,
                        value='days_since_prior_order'
                    ),
                    dcc.RadioItems(
                        id='transformation-type',
                        options=[
                            {'label': 'Log Transformation', 'value': 'log'},
                            {'label': 'Square Root Transformation', 'value': 'sqrt'}
                        ],
                        value='log'
                    ),
                    dcc.Graph(id='transformed-data'),
                ], style={'width': '33%', 'display': 'inline-block', 'border': 'thin lightgrey solid',
                          'padding': '5px'}),
            ], style={'display': 'flex', 'flex-direction': 'row', 'width': '100%', 'margin-bottom': '20px'}),

            html.Div([
                html.Div([
                    dcc.Dropdown(
                        id='xaxis-column',
                        options=dropdown_options,
                        value='order_number'
                    ),
                    dcc.Dropdown(
                        id='yaxis-column',
                        options=dropdown_options,
                        value='days_since_prior_order'
                    ),
                    dcc.Checklist(
                        id='trend-line',
                        options=[{'label': 'Show Trend Line', 'value': 'show'}],
                        value=[]
                    ),
                    dcc.Graph(id='interactive-scatter-plot'),
                    html.Div(id='r-squared-value', style={'font-size': '20px'}),
                ], style={'width': '50%', 'display': 'inline-block', 'border': 'thin lightgrey solid',
                          'padding': '5px'}),

                html.Div([
                    dcc.Dropdown(
                        id='pca-components',
                        options=[{'label': f'{i} Components', 'value': i} for i in range(1, len(numeric_cols) + 1)],
                        value=2
                    ),
                    html.Button('Perform PCA', id='perform-pca-btn', n_clicks=0),
                    dcc.Graph(id='pca-analysis'),
                ], style={'width': '50%', 'display': 'inline-block', 'border': 'thin lightgrey solid',
                          'padding': '5px'}),
            ], style={'display': 'flex', 'flex-direction': 'row', 'width': '100%', 'margin-bottom': '20px'}),

            html.Div([
                dcc.Graph(id='correlation-heatmap'),
            ], style={'width': '100%', 'border': 'thin lightgrey solid', 'padding': '5px'}),
        ])

                         # Add a callback for the third tab graph
@app.callback(
    [Output('gauge-chart', 'figure'),
     Output('pie-chart', 'figure'),
     Output('sales-volume-chart', 'figure'),
     Output('cart-position-chart', 'figure')],
    [Input('product-selector', 'value')]
)
def update_graphs(selected_product):
    filtered_df = df[df['product_name'] == selected_product]

    # Gauge Chart: Reorder Rate
    reorder_rate = filtered_df['reordered'].mean()
    gauge_fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=reorder_rate,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "Reorder Rate"},
        gauge={'axis': {'range': [0, 1]}}
    ))

    # Pie Chart: Sales Distribution by DOW
    sales_dow = filtered_df.groupby('order_dow').size()
    pie_fig = go.Figure(data=[go.Pie(labels=sales_dow.index, values=sales_dow, name='Sales DOW')])
    pie_fig.update_layout(title_text="Sales Distribution by DOW")

    # Sales Volume Over Time
    sales_volume = filtered_df.groupby('order_hour_of_day').size()
    sales_volume_fig = go.Figure(
        data=[go.Scatter(x=sales_volume.index, y=sales_volume.values, mode='lines', name='Sales Volume')])
    sales_volume_fig.update_layout(title_text="Sales Volume Over Time")

    # Average Add-to-Cart Order Position
    avg_cart_position = filtered_df['add_to_cart_order'].mean()
    cart_position_fig = go.Figure(data=[go.Bar(x=[selected_product], y=[avg_cart_position], name='Avg Cart Position')])
    cart_position_fig.update_layout(title_text="Average Add-to-Cart Order Position")

    return gauge_fig, pie_fig, sales_volume_fig, cart_position_fig

# Callbacks for updating the plot
@app.callback(
    Output('subplots-graph', 'figure'),
    [Input('department-dropdown', 'value'), Input('top-n-radio', 'value')]
)
def update_figures(selected_department, top_n):
    # Filter the DataFrame based on the selected department
    filtered_df = df_sliced[df_sliced['department_id'] == selected_department]
    top_products = filtered_df['product_name'].value_counts().nlargest(top_n).index
    top_products_df = filtered_df[filtered_df['product_name'].isin(top_products)]

    # Creating subplots
    fig = make_subplots(rows=2, cols=2, specs=[[{'type': 'pie'}, {'type': 'scatter'}], [{'type': 'scatter'}, {'type': 'scatter'}]], subplot_titles=("Top Products", "Reorder Rate vs. Order Count", "Order Hour Trend", "Orders by Day for Selected Department"))

    # Plot 1: Pie Chart of top products
    top_product_counts = filtered_df['product_name'].value_counts().head(top_n)
    fig.add_trace(go.Pie(labels=top_product_counts.index, values=top_product_counts.values), row=1, col=1)

    # Plot 2: Scatter Plot of reorder rate vs frequency for top products
    reorder_rate = top_products_df.groupby('product_name')['reordered'].mean()
    fig.add_trace(go.Scatter(x=reorder_rate.index, y=reorder_rate.values, mode='markers', marker=dict(size=reorder_rate.values*100), text=reorder_rate.index), row=1, col=2)

    # Plot 3: Line Chart showing trend over time (simulated example: order_hour_of_day)
    order_hour_trend = filtered_df.groupby('order_hour_of_day').size()
    fig.add_trace(go.Scatter(x=order_hour_trend.index, y=order_hour_trend.values, mode='lines'), row=2, col=1)

    # Plot 4: Bar Chart showing orders by day for selected department
    order_day_trend = filtered_df.groupby('order_dow').size()
    fig.add_trace(go.Bar(x=order_day_trend.index, y=order_day_trend.values), row=2, col=2)

    fig.update_layout(height=800, showlegend=False)
    fig.update_xaxes(title_text="Day of the Week", row=2, col=2)
    fig.update_yaxes(title_text="Number of Orders", row=2, col=2)

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
    Input('top-n-radio', 'value')
)
def update_graph(selected_aisle, top_n):
    # Filter the data for the selected aisle
    aisle_data = df_sliced[df_sliced['aisle_id'] == selected_aisle]

    # Determine the number of top products to display
    if top_n == 'Top 5':
        n = 5
    elif top_n == 'Top 10':
        n = 10
    else:
        n = len(aisle_data)

    # Define the four plots
    # Plot 1: Order Frequency in the Aisle (Bar Plot)
    order_counts = aisle_data['product_name'].value_counts().head(n)
    fig1 = go.Bar(x=order_counts.index, y=order_counts.values, name='Order Frequency in the Aisle')

    # Plot 2: Top Reordered Products (Horizontal Bar Plot)
    top_reorders = aisle_data[aisle_data['reordered'] == 1]['product_name'].value_counts().head(n)
    fig2 = go.Bar(x=top_reorders.index, y=top_reorders.values, name='Top Reordered Products')

    # Plot 3: Distribution of Product Prices (Histogram)
    product_prices = aisle_data['add_to_cart_order']
    fig3 = go.Histogram(x=product_prices, name='Distribution of Product Prices')

    # Plot 4: Department Contribution (Pie Chart)
    department_sales = aisle_data.groupby('department_id')['order_id'].count()
    fig4 = go.Pie(labels=department_sales.index, values=department_sales.values, name='Department Contribution')

    # Create subplots
    fig = make_subplots(rows=2, cols=2, subplot_titles=("Order Frequency in the Aisle", "Top Reordered Products", "Distribution of Product Prices", "Department Contribution"), specs=[[{}, {}], [{}, {'type': 'domain'}]])

    # Add each subplot to the figure
    fig.add_trace(go.Bar(x=order_counts.index, y=order_counts.values, name='Order Frequency in the Aisle'), row=1, col=1)
    fig.add_trace(go.Bar(x=top_reorders.index, y=top_reorders.values, name='Top Reordered Products'), row=1, col=2)
    fig.add_trace(go.Histogram(x=product_prices, name='Distribution of Product Prices'), row=2, col=1)
    fig.add_trace(go.Pie(labels=department_sales.index, values=department_sales.values, name='Department Contribution'), row=2, col=2)

    # Update layout and axes
    fig.update_layout(height=1200, width=1200, showlegend=True, title_text="Analysis of Aisles")
    fig.update_xaxes(title_text="Product Names", row=1, col=1)
    fig.update_yaxes(title_text="Order Frequency", row=1, col=1)
    fig.update_xaxes(title_text="Product Names", row=1, col=2)
    fig.update_yaxes(title_text="Top Reordered Products", row=1, col=2)
    fig.update_xaxes(title_text="Add to Cart Order", row=2, col=1)
    fig.update_yaxes(title_text="Distribution of Product Prices", row=2, col=1)

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

    combined_fig = make_subplots(rows=1, cols=2, subplot_titles=(f'{selected_feature} With Outliers', f'{selected_feature} Without Outliers'))
    combined_fig.add_trace(go.Box(y=df[selected_feature], name='With Outliers'), row=1, col=1)
    combined_fig.add_trace(go.Box(y=df_no_outliers[selected_feature], name='Without Outliers'), row=1, col=2)
    combined_fig.update_layout(title_text=f"Comparison of {selected_feature}", showlegend=False)

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
        components = pca.fit_transform(df_standardized)
        explained_var = pca.explained_variance_ratio_
        fig = px.bar(x=[f'PC{i+1}' for i in range(n_components)], y=explained_var[:n_components], labels={'x': 'Principal Components', 'y': 'Variance Explained'}, title=f"PCA Analysis ({n_components} Components)")
        return fig

@app.callback(
    [Output('qq-plot', 'figure'),
     Output('normality-test-result', 'children')],
    [Input('normality-column', 'value'),
     Input('normality-test', 'value'),
     Input('slider', 'value')]
)

def update_graph(selected_feature, test_type, n_value):
    data = df[selected_feature].dropna().sample(n=n_value, replace=False, random_state=1)
    if test_type == 'shapiro':
        stat, p = shapiro(data)
        test_name = "Shapiro-Wilk Test"
    elif test_type == 'ks':
        stat, p = kstest(data, 'norm')
        test_name = "Kolmogorov-Smirnov Test"
    elif test_type == 'anderson':
        result = anderson(data)
        stat = result.statistic
        p = result.significance_level
        test_name = "Anderson-Darling Test"
    else:  # D'Agostino's K-squared test
        stat, p = normaltest(data)
        test_name = "D'Agostino's K-squared Test"

    (osm, osr), (slope, intercept, r) = stats.probplot(data, dist="norm")
    line = go.Scatter(x=osm, y=osr * slope + intercept, mode='lines', name='Theoretical')
    markers = go.Scatter(x=osm, y=osr, mode='markers', name='Data')
    fig = go.Figure(data=[line, markers])
    fig.update_layout(title=f'QQ Plot of {selected_feature}')

    return fig, f"{test_name}: Statistics={stat:.2f}, p-value={p}"

@app.callback(
    Output('transformed-data', 'figure'),
    [Input('transformation-feature', 'value'),
     Input('transformation-type', 'value')]
)
def update_transformed_data(selected_feature, transformation):
    
    if transformation == 'log':
        transformation  = np.log(transformation + 1)
    elif transformation  == 'sqrt':
        transformation  = np.sqrt(transformation )
    fig = px.histogram(transformation , nbins=30, title="Data Transformation")
    return fig

# Precompute the correlation matrix
corr_matrix = df[numeric_cols].corr()

@app.callback(
    Output('correlation-heatmap', 'figure'),
    Input('transformation-type', 'value')
)
def update_heatmap(value):
    # Use the precomputed correlation matrix
    fig = px.imshow(corr_matrix, text_auto=True, labels={'color': "Correlation"}, title="Correlation Heatmap")
    return fig


@app.callback(
    Output('interactive-scatter-plot', 'figure'),
    Output('r-squared-value', 'children'),
    Input('xaxis-column', 'value'),
    Input('yaxis-column', 'value'),
    Input('trend-line', 'value')
)
def update_graph(xaxis_column_name, yaxis_column_name, trend_line_value):
    fig = px.scatter(
        data_frame=data,
        x=xaxis_column_name,
        y=yaxis_column_name,
        title=f"Interactive Plot of {xaxis_column_name} vs. {yaxis_column_name}",
        trendline="ols" if 'show' in trend_line_value else None
    )

    # Get r-squared value if trend line is shown
    r_squared = ""
    if 'show' in trend_line_value:
        results = px.get_trendline_results(fig)
        r_squared = f"R-squared: {results.px_fit_results.iloc[0].rsquared:.3f}"

        # Modify trend line to start from 0
        if 'start_from_zero' in trend_line_value:
            fig.update_traces(
                line=dict(dash='dash'),
                selector=dict(type='scatter', mode='lines')
            )
            fig.update_layout(yaxis=dict(range=[0, fig.data[0].y.max()]))

        # Add R-squared value to the plot
        fig.add_annotation(
            x=0.5,
            y=0.9,
            text=r_squared,
            showarrow=False,
            font=dict(size=12)
        )

    return fig, r_squared

# Run the app
if __name__ == '__main__':
    app.run_server(debug=False)  # Set debug=False for production