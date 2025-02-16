import numpy as np
import yfinance as yf
import plotly.graph_objects as go
import pytz
import tensorflow as tf
import pandas as pd
from keras import Model, Input
from keras.layers import LSTM, Dense, Dropout, Reshape, BatchNormalization
from keras.callbacks import EarlyStopping
from sklearn.preprocessing import MinMaxScaler
import pandas_ta as ta
from datetime import datetime, timedelta
import dask
dask.config.set({'dataframe.query-planning': True})
import os  # Importe o módulo 'os'

class Configuracoes:
    numero_velas_historicas = 60
    horizonte_previsao_candles = 6  # Ênfase no próximo candle + 5, AJUSTÁVEL
    fuso_horario = pytz.timezone('America/Sao_Paulo')
    ativo_padrao = 'BTC-USD'
    modo_operacao = 'simulacao'  # 🚨 Modo de operação: SIMULAÇÃO, AJUSTÁVEL

    mapeamento_intervalos = {
        '1m': '1m', '2m': '2m', '5m': '5m', '15m': '15m',
        '30m': '30m', '60m': '60m', '1h': '1h', '1d': '1d',
        '5d': '5d', '1wk': '1wk', '1mo': '1mo', '3mo': '3mo', '1y': '1y'
    }

    interval_period_map = {
        '1m': '7d',      '2m': '14d',     '5m': '30d',
        '15m': '60d',    '30m': '90d',    '60m': '1y',
        '1h': '1y',      '1d': '5y',      '5d': '10y',
        '1wk': 'max',    '1mo': 'max',    '3mo': 'max', '1y': 'max'
    }

def calcular_sma(df, periodo=20):
    """Calcula a Média Móvel Simples (SMA)."""
    return ta.sma(df['Close'], length=periodo)

def calcular_rsi(df, periodo=14):
    """Calcula o Índice de Força Relativa (RSI)."""
    return ta.rsi(df['Close'], length=periodo)

def calcular_macd(df):
    """Calcula o MACD (Moving Average Convergence Divergence)."""
    if 'Close' not in df.columns:
        print("\033[91m\033[1mERRO: Coluna 'Close' não encontrada no DataFrame de entrada.\033[0m") #Mensagem de erro em vermelho
        return None, None, None  # Retorna None se a coluna não existir

    # VERIFICA SE OS DADOS SÃO NUMÉRICOS
    if not pd.api.types.is_numeric_dtype(df['Close']):
        print("\033[91m\033[1mERRO: A coluna 'Close' contém dados não numéricos.\033[0m") #Mensagem de erro em vermelho
        return None, None, None  # Retorna None se os dados não forem numéricos
    macd_series = ta.macd(df['Close'])
    return macd_series['MACD_12_26_9'], macd_series['MACDh_12_26_9'], macd_series['MACDs_12_26_9']

def obter_unidade_tempo(intervalo):
    """Retorna a unidade de tempo."""
    if intervalo in ['1m', '2m', '5m', '15m', '30m', '60m', '1h']:
        return 'minutos'
    elif intervalo in ['1d', '5d']:
        return 'dias'
    elif intervalo in ['1wk']:
        return 'semanas'
    elif intervalo in ['1mo', '3mo']:
        return 'meses'
    elif intervalo in ['1y']:
        return 'anos'
    return 'desconhecido'

def obter_dados(intervalo):
    """Obtém dados históricos, usando período dinâmico."""
    try:
        periodo = Configuracoes.interval_period_map.get(intervalo, 'max')
        dados_previsoes = yf.download(
            tickers=Configuracoes.ativo_padrao,
            interval=intervalo,
            period=periodo,
            progress=False,
            auto_adjust=True
        )
        if dados_previsoes.empty:
            raise ValueError("Nenhum dado retornado. Verifique o ativo e o intervalo.")
        return dados_previsoes  # Retorna Pandas DataFrame diretamente
    except Exception as e:
        print(f"Erro ao obter dados: {e}")
        return None

def obter_dados_tempo_real_simulado(periodo_simulacao):
    """Simula dados em tempo real, com escolha interativa do intervalo."""
    try:
        print("\nIntervalos de tempo disponíveis:")
        intervalos_disponiveis = list(Configuracoes.mapeamento_intervalos.keys())
        for i, intervalo_opcao in enumerate(intervalos_disponiveis):
            print(f"{i + 1}. {intervalo_opcao}")

        while True:
            escolha_intervalo_index = input("Escolha o número do intervalo desejado: ")
            if escolha_intervalo_index.isdigit():
                escolha_intervalo_index = int(escolha_intervalo_index)
                if 1 <= escolha_intervalo_index <= len(intervalos_disponiveis):
                    intervalo_escolhido = intervalos_disponiveis[escolha_intervalo_index - 1]
                    print(f"Intervalo escolhido: {intervalo_escolhido}")
                    break
                else:
                    print("\033[91m\033[1mOpção inválida. Escolha um número da lista.\033[0m")
            else:
                print("\033[91m\033[1mEntrada inválida. Digite um número.\033[0m")

        intervalo = Configuracoes.mapeamento_intervalos[intervalo_escolhido]
        #print(f"DEBUG: Intervalo para yf.download: {intervalo}") # DEBUG

        dados_historicos = yf.download(
            tickers=Configuracoes.ativo_padrao,
            interval=intervalo,
            period=periodo_simulacao,
            progress=False,
            auto_adjust=True
        )

        # Adiciona os indicadores AQUI, antes do loop
        dados_historicos['SMA'] = calcular_sma(dados_historicos)
        dados_historicos['RSI'] = calcular_rsi(dados_historicos)
        macd, macdh, macds = calcular_macd(dados_historicos)
        dados_historicos['MACD'] = macd
        dados_historicos['MACDh'] = macdh
        dados_historicos['MACDs'] = macds
        dados_historicos.dropna(inplace=True) # Remove NaNs depois de calcular os indicadores

        #print(f"DEBUG: Shape de dados_historicos: {dados_historicos.shape}")  # DEBUG
        if dados_historicos.empty:
            raise ValueError(f"Nenhum dado histórico retornado para o intervalo {intervalo} e período {periodo_simulacao}.")

        #print("DEBUG: Dados históricos para simulação (antes do loop):", dados_historicos.head())  # DEBUG
        #print("DEBUG: Retornando iterador...") # DEBUG
        for index, row in dados_historicos.iterrows():
            yield row

    except Exception as e:
        print(f"Erro ao obter dados simulados em tempo real: {e}")
        return None


def tratar_dados(dados_previsoes_dd):
    """Adiciona indicadores, normaliza e retorna o scaler."""
    dados_previsoes = dados_previsoes_dd #Já é pandas

    dados_previsoes['SMA'] = calcular_sma(dados_previsoes)
    dados_previsoes['RSI'] = calcular_rsi(dados_previsoes)
    macd, macdh, macds = calcular_macd(dados_previsoes)
    dados_previsoes['MACD'] = macd
    dados_previsoes['MACDh'] = macdh
    dados_previsoes['MACDs'] = macds

    dados_previsoes.dropna(inplace=True)
    #print(f"Shape dos dados após tratar_dados (após remoção de NaNs): {dados_previsoes.shape}")  # DEBUG
    #print(f"Tipo de retorno da função tratar_dados: {type((dados_previsoes[['Open', 'High', 'Low', 'Close', 'Volume', 'SMA', 'RSI', 'MACD', 'MACDh', 'MACDs']].values, MinMaxScaler(), dados_previsoes.index))}")  # DEBUG

    normalizador = MinMaxScaler(feature_range=(0, 1))
    dados_normalizados = normalizador.fit_transform(dados_previsoes[['Open', 'High', 'Low', 'Close', 'Volume', 'SMA', 'RSI', 'MACD', 'MACDh', 'MACDs']])
    return dados_normalizados, normalizador, dados_previsoes.index


def preparar_dados_para_inteligencia_artificial(dados_completos, normalizador):
    """Prepara dados usando o normalizador existente."""
    #print(f"Tipo de dados_completos (entrada de preparar_dados_para_ia): {type(dados_completos)}, Shape: {dados_completos.shape}")
    #print(f"Tipo de normalizador (entrada de preparar_dados_para_ia): {type(normalizador)}")

    split_index = int(len(dados_completos) * 0.9)  # 90% TREINAMENTO, 10% VALIDAÇÃO
    train_data = dados_completos[:split_index]
    val_data = dados_completos[split_index:]

    #print(f"Shape de train_data ANTES create_sequences: {train_data.shape}")
    #print(f"Shape de val_data ANTES create_sequences: {val_data.shape}")

    X_train, y_train = create_sequences(train_data)
    X_val, y_val = create_sequences(val_data)

    #print(f"Shape de X_train ANTES do retorno: {X_train.shape}")
    #print(f"Shape de y_train ANTES do retorno: {y_train.shape}")
    #print(f"Shape de X_val ANTES do retorno: {X_val.shape}")
    #print(f"Shape de y_val ANTES do retorno: {y_val.shape}")

    return X_train, y_train, X_val, y_val, normalizador


def create_sequences(data_normalized):
    """Cria sequências de entrada e saída para o LSTM."""
    seq_length = Configuracoes.numero_velas_historicas
    projecoes = Configuracoes.horizonte_previsao_candles
    start_index = seq_length

    if len(data_normalized) < seq_length + projecoes + start_index:
        raise ValueError(f"Dados insuficientes. Mínimo necessário: {seq_length + projecoes + start_index} velas.")

    entradas, saidas = [], []
    for i in range(start_index, len(data_normalized) - seq_length - projecoes + 1):
        sequencia_entrada = data_normalized[i:i + seq_length]
        sequencia_saida = data_normalized[i + seq_length:i + seq_length + projecoes, :5]  # OHLCV
        entradas.append(sequencia_entrada)
        saidas.append(sequencia_saida)

    return np.array(entradas), np.array(saidas)


def construir_modelo_preditivo():
    """Arquitetura LSTM."""
    entrada = Input(shape=(Configuracoes.numero_velas_historicas, 10))

    x = LSTM(128, return_sequences=True, activation='tanh')(entrada)
    x = BatchNormalization(axis=-1)(x)
    x = Dropout(0.3)(x)

    x = LSTM(64, return_sequences=False, activation='tanh')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)

    saida = Dense(Configuracoes.horizonte_previsao_candles * 5)(x)
    saida = Reshape((Configuracoes.horizonte_previsao_candles, 5))(saida)

    modelo = Model(inputs=entrada, outputs=saida)
    modelo.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.01), loss='mse', metrics=['mae'])
    return modelo


def treinar_rede_neural(modelo, X_train, y_train, X_val, y_val):
    """Treina o modelo LSTM."""
    early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

    history = modelo.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=100,
        batch_size=32,
        callbacks=[early_stop],
        verbose=2
    )
    print("\nTreinamento finalizado.")
    print(f"Rede neural treinada. Melhor val_loss: {min(history.history['val_loss'])}")

    return history


def prever_e_visualizar(modelo, X_val, normalizador, intervalo, dados_previsoes_dd, indices_originais):
    """Faz previsões, inverte a normalização e visualiza."""
    previsoes = modelo.predict(X_val)

    # Gráfico das previsões cruas (normalizadas)
    fig_previsoes_cruas = go.Figure()
    for i in range(Configuracoes.horizonte_previsao_candles):
        fig_previsoes_cruas.add_trace(go.Scatter(
            x=np.arange(len(previsoes)),
            y=previsoes[:, i, 3],  # Previsões de fechamento (normalizadas)
            mode='lines',
            name=f'Previsão Crua {i + 1}'
        ))
    fig_previsoes_cruas.update_layout(title='Previsões Cruas (Normalizadas)',
                                      xaxis_title='Tempo',
                                      yaxis_title='Valor Normalizado')
    fig_previsoes_cruas.show()

    # Inverter a normalização *apenas* das previsões de OHLCV (as 5 primeiras colunas)
    previsoes_reshaped = previsoes.reshape(-1, Configuracoes.horizonte_previsao_candles, 5)
    normalizador_ohlcv = MinMaxScaler(feature_range=(0, 1))  # Novo normalizador só para OHLCV
    dados_originais_pandas = dados_previsoes_dd #Já é pandas
    normalizador_ohlcv.fit(dados_originais_pandas[['Open', 'High', 'Low', 'Close', 'Volume']])  # Treina com OHLCV original
    previsoes_invertidas = normalizador_ohlcv.inverse_transform(previsoes_reshaped.reshape(-1, 5)).reshape(previsoes_reshaped.shape) #Inverte a transformação
    unidade_tempo = obter_unidade_tempo(intervalo)

    # --- Gráfico 1: Previsões de Preços (sem suavização) ---
    dados_previsoes_pandas = dados_previsoes_dd
    dados_previsoes_pandas = dados_previsoes_pandas.loc[indices_originais]
    fig = go.Figure()
    for i in range(Configuracoes.horizonte_previsao_candles):
        fig.add_trace(go.Scatter(
            x=np.arange(len(previsoes_invertidas)),
            y=previsoes_invertidas[:, i, 3], #Coluna fechamento
            mode='lines',
            name=f'Previsão {i + 1}'
        ))
    fig.update_layout(title='Previsões de Preços (Sem Suavização)',
                      xaxis_title=f'Tempo ({unidade_tempo})',
                      yaxis_title='Preço')
    fig.show()

    # --- Gráfico 2: Previsões vs Preços Reais (Candlestick, Sem Suavização) ---
    precos_reais = dados_previsoes_pandas['Close'].values[-len(previsoes):]
    datas_reais = dados_previsoes_pandas.index[-len(previsoes):]
    fig = go.Figure()
    fig.add_trace(go.Candlestick(
        x=datas_reais,
        open=dados_previsoes_pandas['Open'].values[-len(previsoes):],
        high=dados_previsoes_pandas['High'].values[-len(previsoes):],
        low=dados_previsoes_pandas['Low'].values[-len(previsoes):],
        close=precos_reais,
        name='Candlestick Real'
    ))
    for i in range(Configuracoes.horizonte_previsao_candles):
        fig.add_trace(go.Scatter(
            x=datas_reais,
            y=previsoes_invertidas[:, i, 3], #Plotando a coluna de fechamento
            mode='lines',
            name=f'Previsão {i + 1}'
        ))
    fig.update_layout(
        title='Previsões vs Preços Reais (Candlestick, Sem Suavização)',
        xaxis_title='Data',
        yaxis_title='Preço (US$)',
        xaxis_rangeslider_visible=False
    )
    fig.show()

    # --- Gráfico 3:  Previsões vs Preços Reais (Linhas, Sem Suavização) ---
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=datas_reais,
        y=precos_reais,
        mode='lines',
        name='Preço Real (Close)'
    ))
    for i in range(Configuracoes.horizonte_previsao_candles):
        fig.add_trace(go.Scatter(
            x=datas_reais,
            y=previsoes_invertidas[:, i, 3], #Plotando a coluna de fechamento
            mode='lines',
            name=f'Previsão {i + 1}'
        ))
    fig.update_layout(
        title='Previsões vs Preços Reais (Linhas, Sem Suavização)',
        xaxis_title='Data',
        yaxis_title='Preço (US$)',
        xaxis_rangeslider_visible=False
    )
    fig.show()


def salvar_modelo(modelo, nome_arquivo):
    """Salva o modelo treinado."""
    modelo.save(nome_arquivo.replace('.h5', '.keras'))
    print(f"Modelo salvo como {nome_arquivo.replace('.h5', '.keras')}.")


def carregar_modelo(nome_arquivo):
    """Carrega um modelo treinado."""
    modelo_carregado = tf.keras.models.load_model(nome_arquivo)
    print(f"Modelo carregado de {nome_arquivo}.")
    return modelo_carregado


def avaliar_modelo(modelo, X_val, y_val):
    """Avalia o desempenho do modelo com dados de validação."""
    perda = modelo.evaluate(X_val, y_val, verbose=0)  # verbose=0 para não poluir o output
    print(f"Perda do modelo na validação: {perda[0]:.4f}") # Acessa o primeiro elemento, que é a perda.


def executar_fluxo_completo():
    """Executa o fluxo completo de análise e previsão."""
    intervalo = '1h'
    dados_previsoes_dd = obter_dados(intervalo)
    dados_normalizados, normalizador, indices_originais = tratar_dados(dados_previsoes_dd) # Recebe Dask DataFrame original
    X_train, y_train, X_val, y_val, normalizador = preparar_dados_para_inteligencia_artificial(dados_normalizados, normalizador)

    modelo = construir_modelo_preditivo()
    treinar_rede_neural(modelo, X_train, y_train, X_val, y_val)

    prever_e_visualizar(modelo, X_val, normalizador, intervalo, dados_previsoes_dd, indices_originais) # Passa Dask DataFrame original e índices

    salvar_modelo(modelo, 'modelo_preditivo_otimizado.keras')
    modelo_carregado = carregar_modelo('modelo_preditivo_otimizado.keras')


def menu_interativo():
    """Menu interativo para o usuário."""
    modelo = None
    X_train, y_train, X_val, y_val, normalizador = None, None, None, None, None
    dados_previsoes_dd = None
    indices_originais = None
    modelo_carregado = None
    intervalo = None

    while True:
        #os.system('cls' if os.name == 'nt' else 'clear')  # Limpa a tela  REMOVIDO
        print("\nMenu Interativo:")
        print("1. Obter Dados e Treinar Modelo (Automático)")  # Opção combinada
        print("2. Simulação de Tempo Real")
        print("3. Carregar Modelo")
        print("4. Salvar Modelo")
        print("5. Avaliar Modelo")
        print("6. Sair")

        escolha = input("Escolha uma opção (1-6): ")

        try:
            if escolha == '1':
                intervalo = input("Intervalo desejado (ex: 1m, 1h, 1d): ")
                dados_previsoes_dd = obter_dados(intervalo)
                if dados_previsoes_dd is not None and dados_previsoes_dd.shape[0] > 0:
                    print(f"\033[92m\033[1mDados obtidos para o intervalo {intervalo}.\033[0m")
                    print(f"\033[92m\033[1mPreparando dados para IA...\033[0m")
                    dados_normalizados, normalizador, indices_originais = tratar_dados(dados_previsoes_dd)
                    X_train, y_train, X_val, y_val, normalizador = preparar_dados_para_inteligencia_artificial(dados_normalizados, normalizador)
                    print(f"\033[92m\033[1mDados preparados.\033[0m")
                    print(f"\033[92m\033[1mConstruindo modelo...\033[0m")
                    modelo = construir_modelo_preditivo()
                    print(f"\033[92m\033[1mModelo construído.\033[0m")
                    print(f"\033[92m\033[1mTreinando rede neural...\033[0m")
                    history = treinar_rede_neural(modelo, X_train, y_train, X_val, y_val)
                    print(f"\033[92m\033[1mRede neural treinada com sucesso.\033[0m")
                    # Agora, as previsões e visualizações são feitas automaticamente após o treinamento.
                    prever_e_visualizar(modelo, X_val, normalizador, intervalo, dados_previsoes_dd, indices_originais)
                    print(f"\033[92m\033[1mPrevisões geradas e visualizadas.\033[0m")

                else:
                    print(f"\033[91m\033[1mNão foi possível obter dados para o intervalo {intervalo}. Verifique o intervalo ou a disponibilidade de dados no Yahoo Finance.\033[0m")
                    dados_previsoes_dd = None

            elif escolha == '2':  # Simulação de Tempo Real
                if modelo is None:
                    print(f"\033[91m\033[1mPor favor, treine o modelo primeiro (Opção 1) antes de iniciar a simulação.\033[0m")
                    continue

                periodo_simulacao = input("Digite o período histórico para simulação (ex: 7d, 30d, 1y): ")
                stream_dados_simulados = obter_dados_tempo_real_simulado(periodo_simulacao)

                if stream_dados_simulados:
                    print("\nIniciando Simulação de Tempo Real...\n")
                    normalizador_simulacao = MinMaxScaler(feature_range=(0, 1))
                    normalizador_ohlcv_simulacao = MinMaxScaler(feature_range=(0, 1))
                    dados_para_normalizar = []
                    #dados_ohlcv_simulacao = []  # Removido

                    # --- TREINAR o normalizador_ohlcv_simulacao ANTES do loop ---
                    # Usar os dados ORIGINAIS do treinamento (OHLCV)
                    dados_treino_pandas = dados_previsoes_dd  # Pega os dados que foram usados no treinamento
                    normalizador_ohlcv_simulacao.fit(dados_treino_pandas[['Open', 'High', 'Low', 'Close', 'Volume']])
                    print("DEBUG: Normalizador OHLCV treinado com dados do TREINAMENTO!")

                    for tick_data in stream_dados_simulados:
                        # --- 1. Preparar os dados para o modelo ---
                        tick_df = pd.DataFrame([tick_data])
                        tick_df = tick_df[['Open', 'High', 'Low', 'Close', 'Volume']]

                        # Adicionar os indicadores
                        tick_df['SMA'] = calcular_sma(tick_df)
                        tick_df['RSI'] = calcular_rsi(tick_df)
                        macd, macdh, macds = calcular_macd(tick_df)
                        tick_df['MACD'] = macd
                        tick_df['MACDh'] = macdh
                        tick_df['MACDs'] = macds
                        tick_df.dropna(inplace=True)

                        if tick_df.empty:
                            continue

                        # --- Normalizar os dados completos (incluindo indicadores) ---
                        dados_para_normalizar.append(tick_df.values)

                        if len(dados_para_normalizar) < Configuracoes.numero_velas_historicas + 1:
                            continue

                        dados_normalizados_sim = normalizador_simulacao.fit_transform(np.concatenate(dados_para_normalizar))

                        # --- 2. Criar sequencias para o modelo LSTM ---
                        X_sim = []
                        sequencia = dados_normalizados_sim[-Configuracoes.numero_velas_historicas:]
                        X_sim.append(sequencia)
                        X_sim = np.array(X_sim)

                        # --- 3. Fazer a previsão ---
                        if X_sim.size == 0 or X_sim.shape[1] < Configuracoes.numero_velas_historicas:
                            continue

                        previsao = modelo.predict(X_sim)

                        # --- 4. Inverter a normalização da previsão (USANDO O NOVO NORMALIZADOR) ---
                        previsao_real = normalizador_ohlcv_simulacao.inverse_transform(previsao.reshape(-1, 5)).reshape(previsao.shape)


                        # --- Visualização em Candlestick (Simulação) ---
                        previsoes_ohlc = previsao_real[0, :, :4]  # Pega as previsões (Open, High, Low, Close)
                        datas_simulacao = [tick_data.name + timedelta(hours=i) for i in range(1, Configuracoes.horizonte_previsao_candles + 1)]

                        fig_sim = go.Figure()
                        fig_sim.add_trace(go.Candlestick(
                            x=datas_simulacao,
                            open=previsoes_ohlc[:, 0],
                            high=previsoes_ohlc[:, 1],
                            low=previsoes_ohlc[:, 2],
                            close=previsoes_ohlc[:, 3],
                            name='Previsão Candlestick'
                        ))

                        fig_sim.update_layout(
                            title='Previsão de Candlestick (Simulação)',
                            xaxis_title='Tempo',
                            yaxis_title='Preço',
                            xaxis_rangeslider_visible=False
                        )
                        fig_sim.show()

                        print(f"[{tick_data.name}] Previsão de fechamento (próximo candle): {previsao_real[0, 0, 3]}")

                        # --- (Futuramente) Aqui entrará a "segunda inteligência" ---
                        dados_para_normalizar.pop(0)  # Remove o dado mais antigo (mantém o tamanho da janela)


                else:
                    print(f"\033[91m\033[1mFalha ao iniciar simulação de tempo real.\033[0m")

            elif escolha == '3':
                nome_arquivo = input("Digite o nome do arquivo para carregar o modelo: ")
                modelo_carregado = carregar_modelo(nome_arquivo)
                print(f"\033[92m\033[1mModelo carregado com sucesso.\033[0m")

            elif escolha == '4':
                if modelo is not None:
                    salvar_modelo(modelo, 'modelo_preditivo_otimizado.keras')
                    print(f"\033[92m\033[1mModelo salvo com sucesso.\033[0m")
                else:
                    print(f"\033[91m\033[1mPor favor, treine o modelo primeiro.\033[0m")

            elif escolha == '5':
                if modelo is not None and X_val is not None and y_val is not None:
                    avaliar_modelo(modelo, X_val, y_val)
                    print(f"\033[92m\033[1mModelo avaliado com sucesso.\033[0m")
                else:
                    print(f"\033[91m\033[1mPor favor, treine o modelo e prepare os dados primeiro.\033[0m")
            elif escolha == '6':
                print("Saindo...")
                break

            else:
                print(f"\033[91m\033[1mOpção inválida.\033[0m")
        except Exception as e:
            print(f"\033[91m\033[1mErro: {e}\033[0m")

if __name__ == "__main__":
    menu_interativo()