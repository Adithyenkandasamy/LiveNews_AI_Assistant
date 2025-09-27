// Enhanced Flash Feed Platform - JavaScript
let chatOpen = false;

function openArticle(articleId) {
    window.location.href = `/article/${articleId}`;
}

function refreshNews() {
    const button = document.querySelector('.refresh-btn');
    const originalText = button.innerHTML;
    button.innerHTML = '⏳ Refreshing...';
    button.disabled = true;
    
    fetch('/api/refresh')
        .then(response => response.json())
        .then(data => {
            if (data.success) {
                location.reload();
            } else {
                alert('Failed to refresh news. Please try again.');
            }
        })
        .catch(error => {
            console.error('Refresh error:', error);
            alert('Network error. Please try again.');
        })
        .finally(() => {
            button.innerHTML = originalText;
            button.disabled = false;
        });
}

function toggleChat() {
    const chatPanel = document.getElementById('chatPanel');
    chatOpen = !chatOpen;
    
    if (chatOpen) {
        chatPanel.style.display = 'block';
        setTimeout(() => chatPanel.classList.add('active'), 10);
    } else {
        chatPanel.classList.remove('active');
        setTimeout(() => chatPanel.style.display = 'none', 300);
    }
}

function sendMessage() {
    const input = document.getElementById('chatInput');
    const message = input.value.trim();
    
    if (!message) return;
    
    // Add user message
    addMessage(message, 'user');
    input.value = '';
    
    // Add typing indicator
    const typingId = addMessage('AI is typing...', 'bot typing');
    
    // Send to backend
    fetch('/api/chat', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({ question: message })
    })
    .then(response => response.json())
    .then(data => {
        document.getElementById(typingId).remove();
        addMessage(data.answer || 'Sorry, I couldn\'t process that.', 'bot');
    })
    .catch(error => {
        document.getElementById(typingId).remove();
        addMessage('Sorry, I encountered an error. Please try again.', 'bot');
        console.error('Chat error:', error);
    });
}

function addMessage(text, sender) {
    const messagesContainer = document.getElementById('chatMessages');
    const messageDiv = document.createElement('div');
    const messageId = 'msg_' + Date.now();
    
    messageDiv.id = messageId;
    messageDiv.className = `message ${sender}`;
    messageDiv.textContent = text;
    
    messagesContainer.appendChild(messageDiv);
    messagesContainer.scrollTop = messagesContainer.scrollHeight;
    
    return messageId;
}

// Auto-refresh stats every 5 minutes
setInterval(() => {
    fetch('/api/stats')
        .then(response => response.json())
        .then(data => {
            updateStats(data);
        })
        .catch(error => console.log('Stats update error:', error));
}, 300000);

function updateStats(stats) {
    // Update stats display dynamically
    const statsBar = document.querySelector('.stats-bar');
    if (statsBar && stats) {
        // Find and update individual stat items
        const statItems = statsBar.querySelectorAll('.stat-item');
        if (statItems.length >= 6) {
            statItems[0].textContent = `📰 ${stats.total_articles} Articles`;
            statItems[1].textContent = `📡 ${stats.sources_count} Sources`;
            statItems[2].textContent = `🏷️ ${stats.categories_count} Categories`;
            statItems[3].textContent = `🤖 ${stats.ai_enhanced} AI Enhanced`;
            statItems[4].textContent = `⏰ Updated ${stats.last_updated}`;
            
            // Update API status
            if (stats.quota_status === 'Limited') {
                statItems[5].innerHTML = '⚠️ API Limited';
                statItems[5].style.background = 'linear-gradient(135deg, #ff7675, #fd79a8)';
            } else {
                statItems[5].innerHTML = '✅ API Active';
                statItems[5].style.background = 'linear-gradient(135deg, #00b894, #00cec9)';
            }
        }
    }
}

// Handle Enter key in chat input
document.addEventListener('DOMContentLoaded', function() {
    const chatInput = document.getElementById('chatInput');
    if (chatInput) {
        chatInput.addEventListener('keypress', function(e) {
            if (e.key === 'Enter') {
                sendMessage();
            }
        });
    }
});
