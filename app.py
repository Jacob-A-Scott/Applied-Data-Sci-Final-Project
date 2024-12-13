import dash
import dash_bootstrap_components as dbc
from dash_bootstrap_templates import load_figure_template
from dash import dcc, html, Input, Output, State
import plotly.graph_objs as go
import pandas as pd
import random

# Lead names (including the composite that I'll create)
LEADS = ['Composite', 'I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']

# Initialize App
app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.LITERA],
    suppress_callback_exceptions=True
)

server=app.server

# Loading theme template
load_figure_template('litera')

df = pd.read_csv('beats_subset.csv')
print("Starting...")
print(f"Dataframe shape: {df.shape}")

# Convert Beat_ID to str for later merge with annotations
df['Beat_ID'] = df['Beat_ID'].astype(str)

unique_beats = df['Beat_ID'].unique()
print("Number of unique beats:", len(unique_beats))

# Create Composite signal by averaging some leads (researched good composites)
# "The combined evaluation of Lead I, Lead II and aVF – allows rapid and accurate QRS assessment."
#  - https://litfl.com/ecg-axis-interpretation/
composite_leads = ['I', 'II', 'aVF']
composite_title = 'Composite (Mean of I, II, aVF)'
print(composite_title)
df['Composite'] = df[composite_leads].mean(axis=1)
print("Composite signal created.")

# App layout using bootstrap to theme it
app.layout = dbc.Container([

    dbc.Row(
        dbc.Col(
            html.H1("QRS Complex Labeling Tool", className='text-center text-primary mt-4 mb-2'),
            width=12
        )
    ),

    dbc.Row(
        dbc.Col(
            html.H5("Labeling Q Onset, R Peak, and S Offset in Single-Beat ECGs", className='text-center mb-4'),
            width=12
        )
    ),

    dbc.Row([
        dbc.Col([
            dbc.Label("Enter the number of beats you'd like to label:"),
            dbc.Input(id='num-beats-input', type='number', value=5, min=1, step=1),
            dbc.Button("Start Labeling", id='start-button', n_clicks=0, color='primary', className='mt-2')
        ], width=4)
    ], justify='center', className='mt-4 mb-4'),

    # This section will expand and become visible when the input beats are defined
    dbc.Collapse([

            dbc.Row([
                    dbc.Col([
                        dbc.Alert(
                        [html.Strong("Tip:"),
                         html.Br(),
                         html.P("Click or double-click legend items to toggle the visibility of leads.")],
                        color="info",
                        dismissable=True,
                        is_open=True,
                        ),
                    ], width=4, className='me-4')
                ], justify='end', className='mt-4'),

        dbc.Card([

            dbc.CardBody([

                html.H6(id='progress-info', className='text-left text-primary'),
                html.H5(id='current-beat-title', className='text-center'),

                # Main ECG plot!
                dcc.Graph(id='ecg-plot', config={'displayModeBar': True}, style={'width': '100%', 'height': '45vh'}),

                html.Br(),

                # QRS boundary selection
                dbc.Label("Select which QRS boundary you're labeling:"),
                dbc.Select(
                    id='boundary-selector',
                    options=[
                        {'label': 'Q Onset', 'value': 'Q_Onset'},
                        {'label': 'R Peak', 'value': 'R_Peak'},
                        {'label': 'S Offset', 'value': 'S_Offset'}
                    ],
                    value='Q_Onset',
                    className='mb-3'
                ),

                # User tip for plot legend
                html.Div("Click on the plot at the correct time point for the selected boundary.", className='mb-3'),

                dbc.Row([
                    dbc.Col([
                        dbc.Label("Q Onset:"),
                        dbc.Input(id='q-onset', type='number', readonly=True)
                    ], width=4),
                    dbc.Col([
                        dbc.Label("R Peak:"),
                        dbc.Input(id='r-peak', type='number', readonly=True)
                    ], width=4),
                    dbc.Col([
                        dbc.Label("S Offset:"),
                        dbc.Input(id='s-offset', type='number', readonly=True)
                    ], width=4),
                ], className='mb-4'),

                # annotation handling buttons
                dbc.Button("Submit Beat Annotations", id='submit-beat-button', color='success', className='me-2'),
                dbc.Button("Skip Beat", id='skip-beat-button', color='secondary'),

                html.Br(),
                html.Br(),

                # Annotated data download button
                dbc.Button("Download Annotations", id='download-button', color='info'),
                dcc.Download(id="download-dataframe-csv")
            ])
        ]),
        ],
        id='labeling-section',
        is_open=False
    ),

    # Hidden stores to keep track of state
    dcc.Store(id='current-beat-index', data=0),
    dcc.Store(id='total-beats-to-label', data=None),
    dcc.Store(id='annotations-store', data={}),
    dcc.Store(id='unique-beats', data=unique_beats.tolist())
], fluid=True)


def load_beat_data(beat_id):
    beat_df = df[df['Beat_ID'] == beat_id].sort_values('Time').copy()
    if not beat_df.empty and 'Composite' not in beat_df.columns:
        beat_df['Composite'] = beat_df[composite_leads].mean(axis=1)
    return beat_df


# Handle start, submit, skip actions
@app.callback(
    Output('current-beat-index', 'data'),
    Output('annotations-store', 'data'),
    Output('total-beats-to-label', 'data'),
    Output('labeling-section', 'is_open'),
    Output('unique-beats', 'data'),
    Input('start-button', 'n_clicks'),
    Input('submit-beat-button', 'n_clicks'),
    Input('skip-beat-button', 'n_clicks'),
    State('current-beat-index', 'data'),
    State('total-beats-to-label', 'data'),
    State('q-onset', 'value'),
    State('r-peak', 'value'),
    State('s-offset', 'value'),
    State('annotations-store', 'data'),
    State('unique-beats', 'data'),
    State('num-beats-input', 'value')
)
def handle_button_actions(start_clicks, submit_clicks, skip_clicks, current_idx, total_beats, q_onset, r_peak, s_offset,
                          annotations, unique_beats_list, num_beats):
    ctx = dash.callback_context
    if not ctx.triggered:
        return current_idx, annotations, total_beats, False, unique_beats_list

    button_id = ctx.triggered[0]['prop_id'].split('.')[0]

    if button_id == 'start-button':
        if num_beats is not None and num_beats > 0:
            # Randomly select n beats
            new_order = random.sample(unique_beats_list, min(num_beats, len(unique_beats_list)))
            return 0, {}, num_beats, True, new_order
        else:
            return dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update

    elif button_id == 'submit-beat-button':
        if None not in (q_onset, r_peak, s_offset):
            if current_idx >= total_beats:
                return current_idx, annotations, total_beats, True, unique_beats_list
            beat_id = unique_beats_list[current_idx]
            annotations[str(beat_id)] = {
                'Q_Onset': q_onset,
                'R_Peak': r_peak,
                'S_Offset': s_offset
            }
            next_idx = current_idx + 1
            return next_idx, annotations, total_beats, True, unique_beats_list
        else:
            return dash.no_update, dash.no_update, dash.no_update, True, dash.no_update

    elif button_id == 'skip-beat-button':
        next_idx = current_idx + 1
        return next_idx, annotations, total_beats, True, unique_beats_list

    return dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update


# Update the displayed beat
@app.callback(
    Output('current-beat-title', 'children'),
    Output('ecg-plot', 'figure'),
    Output('progress-info', 'children'),
    Input('current-beat-index', 'data'),
    State('total-beats-to-label', 'data'),
    State('unique-beats', 'data')
)
def update_beat_display(current_beats_idx, total_beats, unique_beats_list):
    if total_beats is None:
        print("Total beats to label is None.")
        return dash.no_update, dash.no_update, dash.no_update

    if current_beats_idx >= total_beats:
        print("All beats have been labeled.")
        return "All beats have been labeled.", go.Figure(), f"Labeled {total_beats}/{total_beats} beats."

    if current_beats_idx >= len(unique_beats_list):
        print("No more beats available to label.")
        return "No more beats available.", go.Figure(), f"Labeled {current_beats_idx}/{total_beats} beats."

    beat_id = unique_beats_list[current_beats_idx]
    print(f"Currently labeling Beat ID: {beat_id}")

    beat_df = load_beat_data(beat_id)

    if beat_df.empty:
        print(f"Beat ID {beat_id} has no data.")
        return f"Beat ID {beat_id} has no data.", go.Figure(), f"Labeled {current_beats_idx}/{total_beats} beats."

    # Actual plot calls
    fig = go.Figure()
    if 'Composite' in beat_df.columns and beat_df['Composite'].notnull().all():
        fig.add_trace(go.Scatter(
            x=beat_df['Time'],
            y=beat_df['Composite'],
            mode='lines',
            name=composite_title,
            line=dict(width=2, color='black'),
            visible=True,
            hovertemplate='<br>'.join([
                '<b>Composite</b>',
                'Time: %{x:.6f} s',
                'Amplitude: %{y:.6f} mV',
                '<extra></extra>'
            ])
        ))

    for lead in LEADS:
        if lead == 'Composite':
            continue
        if lead in beat_df.columns:
            fig.add_trace(go.Scatter(
                x=beat_df['Time'],
                y=beat_df[lead],
                mode='lines',
                name=lead,
                visible='legendonly',
                hovertemplate='<br>'.join([
                    f'<b>Lead:</b> {lead}',
                    'Time: %{x:.2f} s',
                    'Amplitude: %{y:.2f} mV',
                    '<extra></extra>'  # For some reason, this gets rid of extra characters appearing outside the tooltip.
                                       # Found here: https://stackoverflow.com/questions/60715706/how-to-remove-trace0-here
                ])
            ))
        else:
            print(f"Lead '{lead}' is missing in Beat ID {beat_id}.")

    fig.update_layout(
        xaxis_title='Time (s)',
        yaxis_title='mV',
        legend_title='Leads',
        hovermode='closest',
        template='litera',
        legend=dict(
            font=dict(color="black")
        )
    )

    progress = f"Beat {current_beats_idx + 1}/{total_beats}"
    print(progress)
    return f"Currently Labeling Beat ID: {beat_id}", fig, progress


# Handle boundary updates and resets
@app.callback(
    Output('q-onset', 'value'),
    Output('r-peak', 'value'),
    Output('s-offset', 'value'),
    Input('ecg-plot', 'clickData'),
    Input('current-beat-index', 'data'),
    State('boundary-selector', 'value'),
    State('q-onset', 'value'),
    State('r-peak', 'value'),
    State('s-offset', 'value')
)
def update_boundaries(clickData, current_beats_idx, selected_boundary, q_onset, r_peak, s_offset):
    ctx = dash.callback_context
    if not ctx.triggered:
        return q_onset, r_peak, s_offset

    triggered_input = ctx.triggered[0]['prop_id'].split('.')[0]

    if triggered_input == 'current-beat-index':
        # Reset the annotation input fields when the beat changes
        return None, None, None

    if triggered_input == 'ecg-plot':
        # Updates the annotation field with the value at the click point
        if clickData is not None:
            point_index = clickData['points'][0].get('pointIndex', None)
            if point_index is not None:
                if selected_boundary == 'Q_Onset':
                    q_onset = point_index
                elif selected_boundary == 'R_Peak':
                    r_peak = point_index
                elif selected_boundary == 'S_Offset':
                    s_offset = point_index

    return q_onset, r_peak, s_offset


# Download function
#  includes original data plus binary columns for QRS labels
@app.callback(
    Output("download-dataframe-csv", "data"),
    Input("download-button", "n_clicks"),
    State('annotations-store', 'data'),
    State('unique-beats', 'data'),
    prevent_initial_call=True
)
def download_annotations(n_clicks, annotations, unique_beats_list):
    print("Download callback triggered. n_clicks:", n_clicks)
    if n_clicks is None or n_clicks == 0:
        print("Download button not clicked yet.")
        return None

    if not annotations:
        print("No annotations to download.")
        return None

    annotated_rows = []
    print("Annotations:", annotations)

    for beat_id_str, vals in annotations.items():
        q_idx = vals.get('Q_Onset')
        r_idx = vals.get('R_Peak')
        s_idx = vals.get('S_Offset')

        print(f"Processing Beat_ID: {beat_id_str}, Q_Onset: {q_idx}, R_Peak: {r_idx}, S_Offset: {s_idx}")

        beat_df = df[df['Beat_ID'] == beat_id_str].sort_values('Time').copy()

        if beat_df.empty:
            continue

        beat_df['Q_Onset_flag'] = 0
        beat_df['R_Peak_flag'] = 0
        beat_df['S_Offset_flag'] = 0

        # Set flags if indices are valid
        # i.e. not empty and between 0 and the length of the df [)
        if q_idx is not None and 0 <= q_idx < len(beat_df):
            beat_df.iloc[q_idx, beat_df.columns.get_loc('Q_Onset_flag')] = 1
        if r_idx is not None and 0 <= r_idx < len(beat_df):
            beat_df.iloc[r_idx, beat_df.columns.get_loc('R_Peak_flag')] = 1
        if s_idx is not None and 0 <= s_idx < len(beat_df):
            beat_df.iloc[s_idx, beat_df.columns.get_loc('S_Offset_flag')] = 1

        annotated_rows.append(beat_df)

    annotated_df = pd.concat(annotated_rows, ignore_index=True)
    print("Data prepared for download. Rows:", len(annotated_df))
    return dcc.send_data_frame(annotated_df.to_csv, filename="annotated_beats_with_signals.csv", index=False)


if __name__ == '__main__':
    app.run_server(debug=True)
