document.getElementById('send-button').addEventListener('click', sendMessage);
document.getElementById('message-input').addEventListener('keypress', function(event) {
    if (event.key === 'Enter') {
        event.preventDefault(); // Отменяет действие по умолчанию
        sendMessage();
    }
});

var messageText = '';

function sendMessage() {
    const input = document.getElementById('message-input');
    messageText = input.value.trim();

    if (messageText) {
        const messagesContainer = document.getElementById('messages');
        const newMessage = document.createElement('div');
        newMessage.className = 'message user';
        newMessage.textContent = messageText;

        const url=`http://localhost:5000/api/data/${messageText}`;
        $.get(url, function (data, status) {
            console.log(data.answer[1]);
            const responseMessage = document.createElement('div');
            responseMessage.className = 'message'; // Можно добавить класс для стилизации
            responseMessage.textContent = data.answer[1]; // Получаем ответ
            messagesContainer.appendChild(responseMessage); // Добавляем ответное сообщение
        });

        messagesContainer.appendChild(newMessage);
        input.value = '';
        messagesContainer.scrollTop = messagesContainer.scrollHeight; // Прокрутка вниз

    }
}

fetch(`http://localhost:5000/api/data/${question}`, {
    cache: "no-cache"
})
    .then(response => response.json())
    .then(data => console.log(data))
    .catch(error => console.error('Error:', error));