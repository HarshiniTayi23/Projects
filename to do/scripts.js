let tasks = [];
let currentFilter = 'all';

// Initialize app
document.addEventListener('DOMContentLoaded', function() {
    loadTasks();
    renderTasks();
    
    // Add enter key support
    document.getElementById('taskInput').addEventListener('keypress', function(e) {
        if (e.key === 'Enter') {
            addTask();
        }
    });
});

function addTask() {
    const input = document.getElementById('taskInput');
    const taskText = input.value.trim();
    
    if (taskText === '') {
        alert('Please enter a task!');
        return;
    }

    const task = {
        id: Date.now(),
        text: taskText,
        completed: false,
        createdAt: new Date()
    };

    tasks.unshift(task);
    input.value = '';
    saveTasks();
    renderTasks();
}

function toggleTask(id) {
    const task = tasks.find(t => t.id === id);
    if (task) {
        task.completed = !task.completed;
        saveTasks();
        renderTasks();
    }
}

function deleteTask(id) {
    if (confirm('Are you sure you want to delete this task?')) {
        tasks = tasks.filter(t => t.id !== id);
        saveTasks();
        renderTasks();
    }
}

function filterTasks(filter) {
    currentFilter = filter;
    
    // Update filter buttons
    document.querySelectorAll('.filter-btn').forEach(btn => {
        btn.classList.remove('active');
    });
    event.target.classList.add('active');
    
    renderTasks();
}

function renderTasks() {
    const container = document.getElementById('tasksContainer');
    const emptyState = document.getElementById('emptyState');
    
    let filteredTasks = tasks;
    
    switch (currentFilter) {
        case 'active':
            filteredTasks = tasks.filter(t => !t.completed);
            break;
        case 'completed':
            filteredTasks = tasks.filter(t => t.completed);
            break;
    }

    if (filteredTasks.length === 0) {
        emptyState.style.display = 'block';
        container.innerHTML = '';
        container.appendChild(emptyState);
    } else {
        emptyState.style.display = 'none';
        container.innerHTML = filteredTasks.map(task => `
            <div class="task-item ${task.completed ? 'completed' : ''}">
                <input type="checkbox" class="task-checkbox" 
                       ${task.completed ? 'checked' : ''} 
                       onchange="toggleTask(${task.id})">
                <span class="task-text">${task.text}</span>
                <span class="task-date">${formatDate(task.createdAt)}</span>
                <button class="delete-btn" onclick="deleteTask(${task.id})">Delete</button>
            </div>
        `).join('');
    }

    updateStats();
}

function updateStats() {
    const total = tasks.length;
    const completed = tasks.filter(t => t.completed).length;
    const active = total - completed;
    
    document.getElementById('taskCount').textContent = 
        `${total} total, ${active} active, ${completed} completed`;
}

function formatDate(date) {
    const d = new Date(date);
    return d.toLocaleDateString();
}

function saveTasks() {
    try {
        localStorage.setItem('todoTasks', JSON.stringify(tasks));
    } catch (e) {
        console.log('LocalStorage not available, tasks will not persist');
    }
}

function loadTasks() {
    try {
        const savedTasks = localStorage.getItem('todoTasks');
        if (savedTasks) {
            tasks = JSON.parse(savedTasks);
            // Convert date strings back to Date objects
            tasks.forEach(task => {
                task.createdAt = new Date(task.createdAt);
            });
        } else {
            // Load sample tasks for first-time users
            tasks = [
                {
                    id: 1,
                    text: "Welcome to your new to-do app! 🎉",
                    completed: false,
                    createdAt: new Date()
                },
                {
                    id: 2,
                    text: "Click the checkbox to mark tasks as complete",
                    completed: false,
                    createdAt: new Date()
                },
                {
                    id: 3,
                    text: "Use the filters to view different task types",
                    completed: false,
                    createdAt: new Date()
                }
            ];
        }
    } catch (e) {
        console.log('LocalStorage not available, starting with sample tasks');
        // Fallback sample tasks
        tasks = [
            {
                id: 1,
                text: "Welcome to your new to-do app! 🎉",
                completed: false,
                createdAt: new Date()
            },
            {
                id: 2,
                text: "Click the checkbox to mark tasks as complete",
                completed: false,
                createdAt: new Date()
            },
            {
                id: 3,
                text: "Use the filters to view different task types",
                completed: false,
                createdAt: new Date()
            }
        ];
    }
}