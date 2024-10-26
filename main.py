import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Загрузка датасета из CSV файла с указанием кодировки и пропуском проблемных строк
df = pd.read_csv(r'C:\Python\Dataset\dataset.csv', encoding='latin1', on_bad_lines='skip', sep=';')

# Проверка загруженных данных
print(df.head())

# Проверка имен столбцов
print(df.columns)

# Использование столбцов 'Topic', 'label', и 'Solution'
# Убедитесь, что имена столбцов соответствуют ожидаемым именам
if 'Topic' in df.columns and 'label' in df.columns and 'Solution' in df.columns:
    X = df['Topic']
    y_label = df['label']
    y_solution = df['Solution']
else:
    raise KeyError("Столбцы 'Topic', 'label' и/или 'Solution' не найдены в DataFrame")

# Предобработка данных
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(X).toarray()

label_encoder_label = LabelEncoder()
y_label = label_encoder_label.fit_transform(y_label)

label_encoder_solution = LabelEncoder()
y_solution = label_encoder_solution.fit_transform(y_solution.dropna())  # Удаляем NaN значения

# Сохранение vectorizer и label_encoder в один файл
np.savez('preprocessing.npz',
         vectorizer=vectorizer,
         label_encoder_label=label_encoder_label,
         label_encoder_solution=label_encoder_solution)

# Загрузка vectorizer и label_encoder из файла
loaded_preprocessing = np.load('preprocessing.npz', allow_pickle=True)
vectorizer = loaded_preprocessing['vectorizer'].item()
label_encoder_label = loaded_preprocessing['label_encoder_label'].item()
label_encoder_solution = loaded_preprocessing['label_encoder_solution'].item()

# Удаление строк с пропущенными значениями в столбце 'Solution' для модели решений
df_solution = df.dropna(subset=['Solution'])
X_solution = df_solution['Topic']
y_solution = df_solution['Solution']

# Преобразование данных для модели решений
X_solution = vectorizer.transform(X_solution).toarray()
y_solution = label_encoder_solution.transform(y_solution)

# Разделение данных на обучающую и тестовую выборки для решений
X_train_solution, X_test_solution, y_train_solution, y_test_solution = train_test_split(
    X_solution, y_solution, test_size=0.2, random_state=42)

# Параметры нейронной сети для решений
input_size = X_train_solution.shape[1]
hidden_size = 20
output_size_solution = len(label_encoder_solution.classes_)

weights_input_to_hidden_solution = np.random.uniform(-0.5, 0.5, (hidden_size, input_size))
weights_hidden_to_output_solution = np.random.uniform(-0.5, 0.5, (output_size_solution, hidden_size))
bias_input_to_hidden_solution = np.zeros((hidden_size, 1))
bias_hidden_to_output_solution = np.zeros((output_size_solution, 1))

epochs = 2000
learning_rate = 0.01

# Функция активации (сигмоид)
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Производная сигмоида
def sigmoid_derivative(x):
    return x * (1 - x)

# Обучение нейронной сети для решений
# for epoch in range(epochs):
#     e_loss_solution = 0
#     e_correct_solution = 0

#     for i in range(X_train_solution.shape[0]):
#         image = np.reshape(X_train_solution[i], (-1, 1))
#         label_solution = np.zeros((output_size_solution, 1))
#         label_solution[y_train_solution[i]] = 1

#         # Прямое распространение (к скрытому слою)
#         hidden_raw_solution = bias_input_to_hidden_solution + weights_input_to_hidden_solution @ image
#         hidden_solution = sigmoid(hidden_raw_solution)

#         # Прямое распространение (к выходному слою для решения)
#         output_raw_solution = bias_hidden_to_output_solution + weights_hidden_to_output_solution @ hidden_solution
#         output_solution = sigmoid(output_raw_solution)

#         # Вычисление ошибки для решения
#         e_loss_solution += 1 / len(output_solution) * np.sum((output_solution - label_solution) ** 2, axis=0)
#         e_correct_solution += int(np.argmax(output_solution) == np.argmax(label_solution))

#         # Обратное распространение (выходной слой для решения)
#         delta_output_solution = output_solution - label_solution
#         weights_hidden_to_output_solution += -learning_rate * delta_output_solution @ np.transpose(hidden_solution)
#         bias_hidden_to_output_solution += -learning_rate * delta_output_solution

#         # Обратное распространение (скрытый слой для решения)
#         delta_hidden_solution = np.transpose(weights_hidden_to_output_solution) @ delta_output_solution * sigmoid_derivative(hidden_solution)
#         weights_input_to_hidden_solution += -learning_rate * delta_hidden_solution @ np.transpose(image)
#         bias_input_to_hidden_solution += -learning_rate * delta_hidden_solution

#     # Вывод отладочной информации между эпохами
#     print(f"Epoch №{epoch}")
#     print(f"Loss Solution: {round((e_loss_solution[0] / X_train_solution.shape[0]) * 100, 3)}%")
#     print(f"Accuracy Solution: {round((e_correct_solution / X_train_solution.shape[0]) * 100, 3)}%")
#     e_loss_solution = 0
#     e_correct_solution = 0

# # Сохранение модели для решений
# np.savez('solution_model.npz',
#          weights_input_to_hidden_solution=weights_input_to_hidden_solution,
#          weights_hidden_to_output_solution=weights_hidden_to_output_solution,
#          bias_input_to_hidden_solution=bias_input_to_hidden_solution,
#          bias_hidden_to_output_solution=bias_hidden_to_output_solution)

# Загрузка модели для решений
loaded_solution_model = np.load('solution_model.npz')
weights_input_to_hidden_solution = loaded_solution_model['weights_input_to_hidden_solution']
weights_hidden_to_output_solution = loaded_solution_model['weights_hidden_to_output_solution']
bias_input_to_hidden_solution = loaded_solution_model['bias_input_to_hidden_solution']
bias_hidden_to_output_solution = loaded_solution_model['bias_hidden_to_output_solution']

# Загрузка обученной модели для сервиса
loaded_service_model = np.load('service_model.npz')
weights_input_to_hidden_service = loaded_service_model['weights_input_to_hidden']
weights_hidden_to_output_service = loaded_service_model['weights_hidden_to_output']
bias_input_to_hidden_service = loaded_service_model['bias_input_to_hidden']
bias_hidden_to_output_service = loaded_service_model['bias_hidden_to_output']

# Проверка на тестовых данных для решений
correct_solution = 0
for i in range(X_test_solution.shape[0]):
    image = np.reshape(X_test_solution[i], (-1, 1))
    label_solution = np.zeros((output_size_solution, 1))
    label_solution[y_test_solution[i]] = 1

    # Прямое распространение (к скрытому слою для решения)
    hidden_raw_solution = bias_input_to_hidden_solution + weights_input_to_hidden_solution @ image
    hidden_solution = sigmoid(hidden_raw_solution)

    # Прямое распространение (к выходному слою для решения)
    output_raw_solution = bias_hidden_to_output_solution + weights_hidden_to_output_solution @ hidden_solution
    output_solution = sigmoid(output_raw_solution)

    if np.argmax(output_solution) == np.argmax(label_solution):
        correct_solution += 1

print(f"Test Accuracy Solution: {round((correct_solution / X_test_solution.shape[0]) * 100, 3)}%")

# Пример использования модели для предсказания
def predict(text):
    X_new = vectorizer.transform([text]).toarray()
    X_new = np.reshape(X_new, (-1, 1))

    # Прямое распространение для сервиса
    hidden_raw_service = bias_input_to_hidden_service + weights_input_to_hidden_service @ X_new
    hidden_service = sigmoid(hidden_raw_service)
    output_raw_service = bias_hidden_to_output_service + weights_hidden_to_output_service @ hidden_service
    output_service = sigmoid(output_raw_service)
    predicted_service = label_encoder_label.inverse_transform([np.argmax(output_service)])[0]

    # Прямое распространение для решения
    hidden_raw_solution = bias_input_to_hidden_solution + weights_input_to_hidden_solution @ X_new
    hidden_solution = sigmoid(hidden_raw_solution)
    output_raw_solution = bias_hidden_to_output_solution + weights_hidden_to_output_solution @ hidden_solution
    output_solution = sigmoid(output_raw_solution)
    predicted_solution = label_encoder_solution.inverse_transform([np.argmax(output_solution)])[0]

    return predicted_service, predicted_solution

# Пример предсказания
print(predict("I can't connect to the internet"))
print(predict("The software keeps crashing"))
print(predict("How do I use this feature?"))
print(predict("I haven't received any emails"))