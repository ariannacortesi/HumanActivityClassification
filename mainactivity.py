import os
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
from scipy import stats

DATA_DIR = Path(".") / "FORTH_TRACE_DATASET-master"

# A.2
def participant_data(participant_id):
    participant_folder = f"part{participant_id}"
    full_path = os.path.join(DATA_DIR, participant_folder)
    devices_data = []
    for sensor in range(1, 6):
        file_path = os.path.join(full_path, f"part{participant_id}dev{sensor}.csv")
        if os.path.exists(file_path):
            data_sensor = pd.read_csv(file_path, header=None).to_numpy()
            devices_data.append(data_sensor)
    if len(devices_data) > 0:
        return np.vstack(devices_data)
    return np.empty((0, 12))

def todos_participantes():
    all_data = [participant_data(pid) for pid in range(15)]
    return np.vstack(all_data)

def modulo(matriz, col):
    return np.sqrt(np.sum(matriz[:, col:col+3]**2, axis=1))

# A.3.1
def pergunta_3_1(full_matrix):
    acc_magnitude = modulo(full_matrix, 1)
    activity_labels = full_matrix[:, 11].astype(int)

    data_by_activity = []
    labels = []
    for act in range(1, 17):
        valores = acc_magnitude[activity_labels == act]
        if len(valores) > 0:
            data_by_activity.append(valores)
            labels.append(str(act))

    plt.figure(figsize=(10, 5))
    plt.boxplot(data_by_activity, showfliers=True)
    plt.xticks(range(1, len(labels) + 1), labels)
    plt.title("A.3.1 - Boxplot do Módulo da Aceleração por Atividade")
    plt.xlabel("Atividade")
    plt.ylabel("Módulo da Aceleração")
    plt.show()

# A.3.2
def pergunta_3_2(full_matrix):
    acc_magnitude = modulo(full_matrix, 1)
    activity_labels = full_matrix[:, 11].astype(int)

    print("Densidade de Outliers por Atividade (A.3.2):")
    for act in range(1, 17):
        valores = acc_magnitude[activity_labels == act]
        nr = len(valores)
        if nr > 0:
            q25, q75 = np.percentile(valores, [25, 75])
            iqr = q75 - q25
            outliers = (valores < q25 - 1.5 * iqr) | (valores > q75 + 1.5 * iqr)
            densidade = (np.sum(outliers) / nr) * 100
            print(f"Atividade {act:2d}: {densidade:.2f}%")

# A.3.3
def outliers_zscore(amostras, k):
    media = np.mean(amostras)
    desvio = np.std(amostras, ddof=1)
    return np.abs((amostras - media) / desvio) > k

# A.3.4
def pergunta_3_4(full_matrix):
    acc_magnitude = modulo(full_matrix, 1)
    indices = np.arange(len(acc_magnitude))

    for k in [3, 3.5, 4]:
        outliers = outliers_zscore(acc_magnitude, k)
        print(f"Z-Score k={k}: {np.sum(outliers) / len(acc_magnitude) * 100:.2f}% outliers")
        plt.figure(figsize=(10, 4))
        plt.scatter(indices[~outliers], acc_magnitude[~outliers], color="blue", s=1, label="Normal")
        plt.scatter(indices[outliers], acc_magnitude[outliers], color="red", s=3, label="Outlier")
        plt.title(f"A.3.4 - Outliers Z-Score (k = {k})")
        plt.xlabel("Índice da Amostra")
        plt.ylabel("Módulo da Aceleração")
        plt.legend()
        plt.show()

# A.3.6
def kmeans(dados, n, max_iter=30, seed=42):
    rng = np.random.default_rng(seed)
    centros = dados[rng.choice(len(dados), size=n, replace=False)].copy()
    grupos = np.zeros(len(dados), dtype=int)

    for _ in range(max_iter):
        distancias = np.linalg.norm(dados[:, np.newaxis] - centros, axis=2)
        novos_grupos = np.argmin(distancias, axis=1)
        if np.array_equal(grupos, novos_grupos):
            break
        grupos = novos_grupos
        for i in range(n):
            pontos = dados[grupos == i]
            if len(pontos) > 0:
                centros[i] = np.mean(pontos, axis=0)
    return centros, grupos

# A.3.7
def pergunta_3_7(full_matrix, k=3):
    sub = full_matrix[::20]
    dados = np.column_stack((modulo(sub, 1), modulo(sub, 4), modulo(sub, 7)))
    # normalizar porque acc, gyro e mag têm escalas muito diferentes
    dados_norm = (dados - np.mean(dados, axis=0)) / np.std(dados, axis=0)

    # metodo do cotovelo
    erros = []
    for n in range(1, 11):
        centros, grupos = kmeans(dados_norm, n)
        erros.append(np.sum((dados_norm - centros[grupos])**2))

    plt.figure(figsize=(7, 4))
    plt.plot(range(1, 11), erros, marker="o")
    plt.title("A.3.7 - Método do Cotovelo")
    plt.xlabel("Número de clusters")
    plt.ylabel("Soma dos erros quadráticos")
    plt.show()

    centros, grupos = kmeans(dados_norm, k)

    # outlier = ponto muito longe do centro do seu cluster
    dist = np.linalg.norm(dados_norm - centros[grupos], axis=1)
    outliers = np.zeros(len(dados_norm), dtype=bool)
    for i in range(k):
        d = dist[grupos == i]
        if len(d) > 1:
            outliers[grupos == i] = d > np.mean(d) + 3 * np.std(d)

    print(f"Outliers K-means (k={k}): {np.sum(outliers) / len(dados) * 100:.2f}%")
    print(f"Outliers Z-Score acc (k=3): {np.sum(outliers_zscore(dados[:, 0], 3)) / len(dados) * 100:.2f}%")

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(projection="3d")
    ax.scatter(dados[~outliers, 0], dados[~outliers, 1], dados[~outliers, 2], c=grupos[~outliers], s=2, cmap="viridis")
    ax.scatter(dados[outliers, 0], dados[outliers, 1], dados[outliers, 2], color="red", s=10, label="Outliers")
    ax.set_title(f"A.3.7 - Outliers com K-means (k = {k})")
    ax.set_xlabel("Módulo Acc")
    ax.set_ylabel("Módulo Gyro")
    ax.set_zlabel("Módulo Mag")
    ax.legend()
    plt.show()

# A.3.7.1
def estudo_dbscan_3_7_1(full_matrix):
    sub = full_matrix[::50]
    dados = np.column_stack((modulo(sub, 1), modulo(sub, 4), modulo(sub, 7)))
    dados_norm = (dados - np.mean(dados, axis=0)) / np.std(dados, axis=0)

    modelo = DBSCAN(eps=0.5, min_samples=10)
    rotulos = modelo.fit_predict(dados_norm)
    outliers = rotulos == -1

    densidade = (np.sum(outliers) / len(dados)) * 100
    print(f"Densidade de outliers DBSCAN (A.3.7.1): {densidade:.2f}%")

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(projection="3d")
    ax.scatter(dados[~outliers, 0], dados[~outliers, 1], dados[~outliers, 2], c="blue", s=3, label="Normais")
    ax.scatter(dados[outliers, 0], dados[outliers, 1], dados[outliers, 2], c="red", s=15, label="Outliers")
    ax.set_title("A.3.7.1 - Deteção de Outliers com DBSCAN")
    ax.set_xlabel("Módulo Acc")
    ax.set_ylabel("Módulo Gyro")
    ax.set_zlabel("Módulo Mag")
    ax.legend()
    plt.show()

# A.3.8
def inject_outliers(amostras, x, k, z):
    dados = np.copy(amostras)
    nr = len(dados)

    is_outlier = outliers_zscore(dados, k)
    d = (np.sum(is_outlier) / nr) * 100.0
    escolhidos = np.array([], dtype=int)

    if d < x:
        indices_normais = np.where(~is_outlier)[0]
        qtd = int(np.ceil(((x - d) / 100.0) * nr))
        escolhidos = np.random.choice(indices_normais, size=qtd, replace=False)

        media = np.mean(dados)
        desvio = np.std(dados, ddof=1)

        for idx in escolhidos:
            s = np.random.choice([-1, 1])
            q = np.random.uniform(0, z)
            dados[idx] = media + s * k * (desvio + q)

    # devolvo tambem os indices para saber quais foram alterados no 3.10
    return dados, np.sort(escolhidos)

# A.3.9
def linear_model(X, y):
    X_completo = np.hstack((np.ones((len(X), 1)), X))
    return np.linalg.pinv(X_completo.T @ X_completo) @ X_completo.T @ y

def criar_janelas_autoregressivas(sinal, p):
    n = len(sinal) - p
    X = np.zeros((n, p))
    y = np.zeros(n)
    for i in range(p, len(sinal)):
        X[i - p, :] = sinal[i - p:i]
        y[i - p] = sinal[i]
    return X, y

# A.3.10
def pergunta_3_10():
    dados_p = participant_data(1)
    dados_p = dados_p[dados_p[:, 0] == 1]   # so o dispositivo 1
    acc_magnitude = modulo(dados_p, 1)[:5000]

    np.random.seed(0)
    modulo_outliers, idx_out = inject_outliers(acc_magnitude, 5, 3, 2)

    melhor_p, menor_mse = None, float("inf")
    melhor_rec = None

    for p in [2, 3, 5, 10, 15, 20]:
        X, y = criar_janelas_autoregressivas(acc_magnitude, p)
        beta = linear_model(X, y)

        # substituir os outliers um a um, usando os valores ja corrigidos
        modulo_rec = np.copy(modulo_outliers)
        idx = idx_out[idx_out >= p]
        for pos in idx:
            janela = modulo_rec[pos - p:pos]
            modulo_rec[pos] = beta[0] + janela @ beta[1:]

        mse = np.mean((acc_magnitude[idx] - modulo_rec[idx])**2)
        print(f"p = {p}: MSE = {mse:.4f}")

        if mse < menor_mse:
            menor_mse, melhor_p, melhor_rec = mse, p, modulo_rec

    print(f"A.3.10 - Melhor ordem p: {melhor_p} (MSE: {menor_mse:.4f})")

    idx = idx_out[idx_out >= melhor_p]
    erros = acc_magnitude[idx] - melhor_rec[idx]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].hist(erros, bins=30, edgecolor="black")
    axes[0].set_title("Distribuição do Erro de Previsão")
    axes[0].set_xlabel("Erro (Real - Previsto)")
    axes[0].set_ylabel("Frequência")

    meio = idx[len(idx) // 2]
    ini, fim = max(0, meio - 30), meio + 30
    axes[1].plot(range(ini, fim), acc_magnitude[ini:fim], label="Real", color="blue")
    axes[1].plot(range(ini, fim), melhor_rec[ini:fim], label="Reconstruído", linestyle="--", color="red")
    axes[1].set_title("Aceleração Real vs Prevista")
    axes[1].set_xlabel("Amostra")
    axes[1].set_ylabel("Módulo da Aceleração")
    axes[1].legend()
    plt.tight_layout()
    plt.show()

# A.4.1
def pergunta_4_1(full_matrix):
    acc_magnitude = modulo(full_matrix, 1)
    activity_labels = full_matrix[:, 11].astype(int)

    grupos = []
    for act in range(1, 17):
        valores = acc_magnitude[activity_labels == act]
        grupos.append(valores)

        # normalizar e testar se segue uma distribuicao normal
        z = (valores - np.mean(valores)) / np.std(valores)
        estatistica, p = stats.kstest(z, "norm")
        print(f"Atividade {act}: media = {np.mean(valores):.3f}, p = {p:.4f}")

    # comparar todas as atividades
    h, p = stats.kruskal(*grupos)
    print(f"Kruskal-Wallis: p = {p:.4f}")

# A.4.2 - features do artigo de Zhang & Sawchuk (2011)
FS = 51.2               # frequencia de amostragem (Hz)
JANELA = int(2 * FS)    # janela de 2 segundos = 102 amostras
PASSO = JANELA // 2     # 50% de sobreposicao

#  Features estatisticas (Tabela 1), calculadas em cada eixo 
def f_media(s):
    return np.mean(s)

def f_mediana(s):
    return np.median(s)

def f_desvio(s):
    return np.std(s)

def f_variancia(s):
    return np.var(s)

def f_rms(s):
    return np.sqrt(np.mean(s**2))

def f_derivada_media(s):
    return np.mean(np.diff(s))

def f_assimetria(s):
    dp = np.std(s)
    if dp == 0:
        return 0
    return np.mean((s - np.mean(s))**3) / dp**3

def f_curtose(s):
    dp = np.std(s)
    if dp == 0:
        return 0
    return np.mean((s - np.mean(s))**4) / dp**4 - 3

def f_iqr(s):
    q25, q75 = np.percentile(s, [25, 75])
    return q75 - q25

def f_cruzamento_zero(s):
    return np.sum(np.diff(np.sign(s)) != 0) / len(s)

def f_cruzamento_media(s):
    return f_cruzamento_zero(s - np.mean(s))

def f_correlacao(s1, s2):
    if np.std(s1) == 0 or np.std(s2) == 0:
        return 0
    return np.corrcoef(s1, s2)[0, 1]

def f_entropia_espectral(s):
    potencia = np.abs(np.fft.rfft(s))**2
    if np.sum(potencia) == 0:
        return 0
    p = potencia / np.sum(potencia)
    p = p[p > 0]
    return -np.sum(p * np.log2(p))

#  Features fisicas (secao 3.3)
def f_ai(ax, ay, az):
    return np.mean(np.sqrt(ax**2 + ay**2 + az**2))

def f_vi(ax, ay, az):
    return np.var(np.sqrt(ax**2 + ay**2 + az**2))

def f_sma(ax, ay, az):
    return (np.sum(np.abs(ax)) + np.sum(np.abs(ay)) + np.sum(np.abs(az))) / len(ax)

def f_freq_dominante(s):
    potencia = np.abs(np.fft.rfft(s))**2
    freqs = np.fft.rfftfreq(len(s), 1 / FS)
    return freqs[np.argmax(potencia[1:]) + 1]      

def f_energia(s):
    potencia = np.abs(np.fft.rfft(s))**2
    return np.sum(potencia[1:]) / len(s)           

def f_aae(ax, ay, az):
    return (f_energia(ax) + f_energia(ay) + f_energia(az)) / 3

def f_are(gx, gy, gz):
    return (f_energia(gx) + f_energia(gy) + f_energia(gz)) / 3


def vetor_features(janela):
    # janela: 102 linhas x 6 colunas (ax, ay, az, gx, gy, gz)
    ax, ay, az = janela[:, 0], janela[:, 1], janela[:, 2]
    gx, gy, gz = janela[:, 3], janela[:, 4], janela[:, 5]

    v = []
    for c in range(6):
        s = janela[:, c]
        v += [f_media(s), f_mediana(s), f_desvio(s), f_variancia(s), f_rms(s),
              f_derivada_media(s), f_assimetria(s), f_curtose(s), f_iqr(s),
              f_cruzamento_zero(s), f_cruzamento_media(s), f_entropia_espectral(s),
              f_freq_dominante(s), f_energia(s)]
    for i in range(6):
        for j in range(i + 1, 6):
            v.append(f_correlacao(janela[:, i], janela[:, j]))

    v += [f_ai(ax, ay, az), f_vi(ax, ay, az), f_sma(ax, ay, az),
          f_aae(ax, ay, az), f_are(gx, gy, gz)]
    return v


def nomes_features():
    eixos = ["ax", "ay", "az", "gx", "gy", "gz"]
    stat = ["media", "mediana", "desvio", "variancia", "rms", "derivada",
            "assimetria", "curtose", "iqr", "zcr", "mcr", "entropia",
            "freq_dom", "energia"]
    nomes = []
    for e in eixos:
        for f in stat:
            nomes.append(f"{e}_{f}")
    for i in range(6):
        for j in range(i + 1, 6):
            nomes.append(f"corr_{eixos[i]}_{eixos[j]}")
    nomes += ["AI", "VI", "SMA", "AAE", "ARE"]
    return nomes


def pergunta_4_2():
    linhas = []
    info = []
    for pid in range(15):
        dados = participant_data(pid)
        for dev in range(1, 6):
            d = dados[dados[:, 0] == dev]
            rotulos = d[:, 11].astype(int)

            # dividir em blocos com a mesma atividade
            mudancas = np.where(rotulos[1:] != rotulos[:-1])[0] + 1
            limites = [0] + list(mudancas) + [len(rotulos)]

            for k in range(len(limites) - 1):
                for ini in range(limites[k], limites[k + 1] - JANELA + 1, PASSO):
                    janela = d[ini:ini + JANELA, 1:7]
                    linhas.append(vetor_features(janela))
                    # instante = timestamp do fim da janela
                    info.append([pid, dev, d[ini + JANELA - 1, 10], rotulos[ini]])

    tabela = pd.DataFrame(linhas, columns=nomes_features())
    tabela[["participante", "dispositivo", "timestamp", "atividade"]] = info
    print(f"A.4.2 - {len(tabela)} janelas, {len(nomes_features())} features")
    tabela.to_csv("features_4_2.csv", index=False)
    return tabela





def main():
    full_matrix = todos_participantes()
    #pergunta_3_1(full_matrix)
    
# A.4.3 - PCA
def pca(X):
    # 1) normalizar cada feature com z-score (media 0, desvio 1)
    media = np.mean(X, axis=0)
    desvio = np.std(X, axis=0)
    desvio[desvio == 0] = 1          # evitar divisao por zero
    Xn = (X - media) / desvio

    # 2) matriz de covariancia das features
    cov = np.cov(Xn, rowvar=False)

    # 3) valores e vetores proprios (eigh porque a matriz e simetrica)
    valores, vetores = np.linalg.eigh(cov)

    # 4) ordenar do maior para o menor valor proprio
    ordem = np.argsort(valores)[::-1]
    valores = valores[ordem]
    vetores = vetores[:, ordem]
    return valores, vetores, media, desvio


def pergunta_4_3():
    tabela = pd.read_csv("features_4_2.csv")     # criado no 4.2
    X = tabela[nomes_features()].to_numpy()

    valores, vetores, media, desvio = pca(X)
    print(f"A.4.3 - PCA com {X.shape[1]} features e {X.shape[0]} janelas")
    print("5 maiores valores proprios:", np.round(valores[:5], 3))
    return tabela, valores, vetores, media, desvio
# A.4.4 - importancia de cada vetor proprio
def pergunta_4_4():
    tabela, valores, vetores, media, desvio = pergunta_4_3()

    # percentagem da variancia explicada por cada componente
    percentagem = valores / np.sum(valores) * 100
    acumulada = np.cumsum(percentagem)

    print("Componente | % variancia | % acumulada")
    for i in range(10):
        print(f"{i + 1:10d} | {percentagem[i]:11.2f} | {acumulada[i]:11.2f}")

    # numero de componentes para explicar 75%
    n = np.argmax(acumulada >= 75) + 1
    print(f"Sao precisas {n} componentes para explicar 75% da variancia ({acumulada[n - 1]:.2f}%)")

    plt.figure(figsize=(8, 4))
    plt.bar(range(1, len(percentagem) + 1), percentagem, label="Individual")
    plt.plot(range(1, len(acumulada) + 1), acumulada, color="red", marker=".", label="Acumulada")
    plt.axhline(75, color="gray", linestyle="--", label="75%")
    plt.title("A.4.4 - Variancia explicada por cada componente")
    plt.xlabel("Componente principal")
    plt.ylabel("Variancia explicada (%)")
    plt.legend()
    plt.show()

    # A.4.4.1 - obter as features comprimidas para um instante
    # 1) normalizar a janela com a mesma media e desvio usados no PCA
    # 2) multiplicar pelos n primeiros vetores proprios
    i = 0                                   # instante escolhido (primeira janela)
    x = tabela[nomes_features()].to_numpy()[i]
    x_norm = (x - media) / desvio
    x_pca = x_norm @ vetores[:, :n]

    print(f"\nA.4.4.1 - Instante escolhido: participante {int(tabela['participante'][i])}, "
          f"dispositivo {int(tabela['dispositivo'][i])}, atividade {int(tabela['atividade'][i])}")
    print(f"Vetor original: {len(x)} features -> vetor comprimido: {len(x_pca)} features")
    print("Features comprimidas:", np.round(x_pca, 3))
# A.4.5 - Fisher Score e ReliefF
def fisher_score(X, y):
    # para cada feature: variancia entre classes / variancia dentro das classes
    media_total = np.mean(X, axis=0)
    entre = np.zeros(X.shape[1])
    dentro = np.zeros(X.shape[1])
    for c in np.unique(y):
        Xc = X[y == c]
        nc = len(Xc)
        entre += nc * (np.mean(Xc, axis=0) - media_total)**2
        dentro += nc * np.var(Xc, axis=0)
    dentro[dentro == 0] = 1e-12        # evitar divisao por zero
    return entre / dentro


def relieff(X, y, n_amostras=300, k=10, seed=42):
    # 1) normalizar cada feature para [0, 1]
    minimo = np.min(X, axis=0)
    amplitude = np.max(X, axis=0) - minimo
    amplitude[amplitude == 0] = 1
    Xn = (X - minimo) / amplitude

    classes, contagens = np.unique(y, return_counts=True)
    prob = contagens / len(y)          # probabilidade de cada classe

    pesos = np.zeros(X.shape[1])
    rng = np.random.default_rng(seed)
    escolhidos = rng.choice(len(Xn), n_amostras, replace=False)

    for i in escolhidos:
        # distancia (Manhattan) do ponto i a todos os outros
        dist = np.sum(np.abs(Xn - Xn[i]), axis=1)
        dist[i] = np.inf                # nao contar o proprio ponto

        for c, p in zip(classes, prob):
            idx = np.where(y == c)[0]
            vizinhos = idx[np.argsort(dist[idx])[:k]]     # k vizinhos mais proximos da classe c
            diferenca = np.mean(np.abs(Xn[vizinhos] - Xn[i]), axis=0)

            if c == y[i]:
                pesos -= diferenca      # mesma classe: a feature deve ser parecida
            else:
                p_i = prob[classes == y[i]][0]
                pesos += p / (1 - p_i) * diferenca   # outra classe: a feature deve ser diferente

    return pesos / n_amostras


def pergunta_4_5():
    tabela = pd.read_csv("features_4_2.csv")
    X = tabela[nomes_features()].to_numpy()
    y = tabela["atividade"].to_numpy()

    fisher = fisher_score(X, y)
    relief = relieff(X, y)
    print("A.4.5 - Fisher Score e ReliefF calculados para", len(fisher), "features")
    return fisher, relief


# A.4.6 - 10 melhores features segundo o Fisher Score e o ReliefF
def pergunta_4_6():
    fisher, relief = pergunta_4_5()
    nomes = np.array(nomes_features())

    top_fisher = np.argsort(fisher)[::-1][:10]
    top_relief = np.argsort(relief)[::-1][:10]

    print("\nA.4.6 - 10 melhores features (Fisher Score):")
    for pos, j in enumerate(top_fisher, start=1):
        print(f"{pos:2d}. {nomes[j]:20s} {fisher[j]:.4f}")

    print("\nA.4.6 - 10 melhores features (ReliefF):")
    for pos, j in enumerate(top_relief, start=1):
        print(f"{pos:2d}. {nomes[j]:20s} {relief[j]:.4f}")

    comuns = set(nomes[top_fisher]) & set(nomes[top_relief])
    print("\nFeatures que aparecem nos dois:", sorted(str(c) for c in comuns))

    # A.4.6.1 - obter as features comprimidas para um instante
    # basta ficar so com as colunas das 10 melhores features (valores originais)
    tabela = pd.read_csv("features_4_2.csv")
    i = 0                                   # o mesmo instante do 4.4.1
    x = tabela[nomes_features()].to_numpy()[i]
    print(f"\nA.4.6.1 - Instante: participante {int(tabela['participante'][i])}, "
          f"dispositivo {int(tabela['dispositivo'][i])}, atividade {int(tabela['atividade'][i])}")
    print("Vetor comprimido (Fisher):")
    for j in top_fisher:
        print(f"   {nomes[j]:20s} {x[j]:.3f}")
    print("Vetor comprimido (ReliefF):")
    for j in top_relief:
        print(f"   {nomes[j]:20s} {x[j]:.3f}")
def main():
    full_matrix = todos_participantes()
    pergunta_3_1(full_matrix)
    pergunta_3_2(full_matrix)
    pergunta_3_4(full_matrix)
    pergunta_3_7(full_matrix, k=3)
    estudo_dbscan_3_7_1(full_matrix)
    pergunta_3_10()
    pergunta_4_1(full_matrix)
    pergunta_4_2()
    pergunta_4_3()
    pergunta_4_4()
    pergunta_4_6()


if __name__ == "__main__":
    main()
