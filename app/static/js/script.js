document.addEventListener('DOMContentLoaded', function() {
    const askButton = document.getElementById('ask-button');
    const questionInput = document.getElementById('question');
    const answerDiv = document.getElementById('answer');
    const sourcesDiv = document.getElementById('sources');
    const knowledgeBaseEntries = document.getElementById('knowledge-base-entries');
    
    // Load Knowledge Base data
    loadKnowledgeBase();
    
    // Function to load knowledge base data
    function loadKnowledgeBase() {
        fetch('/api/knowledge-base')
            .then(response => {
                if (!response.ok) {
                    throw new Error('Failed to load knowledge base');
                }
                return response.json();
            })
            .then(data => {
                renderKnowledgeBase(data);
            })
            .catch(error => {
                console.error('Error:', error);
                knowledgeBaseEntries.innerHTML = `<tr><td colspan="4">Failed to load knowledge base. Please refresh the page.</td></tr>`;
            });
    }
    
    // Function to render knowledge base entries
    function renderKnowledgeBase(entries) {
        if (!entries || entries.length === 0) {
            knowledgeBaseEntries.innerHTML = `<tr><td colspan="4" class="empty-kb-message">No documents in knowledge base. Upload a PDF to get started.</td></tr>`;
            return;
        }
        
        // Sort entries by date, newest first
        entries.sort((a, b) => new Date(b.date_added) - new Date(a.date_added));
        
        const rows = entries.map(entry => {
            // Format the file size
            const fileSize = formatFileSize(entry.file_size_bytes);
            
            // Format the date
            let formattedDate;
            try {
                const date = new Date(entry.date_added);
                formattedDate = date.toLocaleString();
            } catch (e) {
                formattedDate = "Unknown date";
            }
            
            return `
                <tr data-id="${entry.id}">
                    <td>${entry.file_name}</td>
                    <td class="file-size">${fileSize}</td>
                    <td>${formattedDate}</td>
                    <td class="file-actions">
                        <button class="delete-btn" data-id="${entry.id}">Remove</button>
                    </td>
                </tr>
            `;
        }).join('');
        
        knowledgeBaseEntries.innerHTML = rows;
        
        // Add event listeners to delete buttons
        document.querySelectorAll('.delete-btn').forEach(button => {
            button.addEventListener('click', function() {
                const fileId = parseInt(this.getAttribute('data-id'), 10);
                deleteKnowledgeBaseEntry(fileId);
            });
        });
    }
    
    // Function to format file size
    function formatFileSize(bytes) {
        if (!bytes || isNaN(bytes)) return "Unknown";
        
        if (bytes < 1024) {
            return bytes + ' B';
        } else if (bytes < 1024 * 1024) {
            return (bytes / 1024).toFixed(1) + ' KB';
        } else if (bytes < 1024 * 1024 * 1024) {
            return (bytes / (1024 * 1024)).toFixed(1) + ' MB';
        } else {
            return (bytes / (1024 * 1024 * 1024)).toFixed(1) + ' GB';
        }
    }
    
    // Function to delete a knowledge base entry
    function deleteKnowledgeBaseEntry(fileId) {
        if (!confirm('Are you sure you want to remove this document from the knowledge base?')) {
            return;
        }
        
        const button = document.querySelector(`.delete-btn[data-id="${fileId}"]`);
        if (button) {
            button.disabled = true;
            button.textContent = "Removing...";
        }
        
        fetch(`/api/knowledge-base/${fileId}`, {
            method: 'DELETE',
        })
        .then(response => {
            if (!response.ok) {
                throw new Error('Failed to delete entry');
            }
            return response.json();
        })
        .then(data => {
            if (data.success) {
                loadKnowledgeBase(); // Reload the knowledge base
            } else {
                alert('Failed to delete the document');
                if (button) {
                    button.disabled = false;
                    button.textContent = "Remove";
                }
            }
        })
        .catch(error => {
            console.error('Error:', error);
            alert(`Error: ${error.message}`);
            if (button) {
                button.disabled = false;
                button.textContent = "Remove";
            }
        });
    }
    
    // Question answering functionality
    askButton.addEventListener('click', async function() {
        askQuestion();
    });
    
    // Function to ask a question
    async function askQuestion() {
        const question = questionInput.value.trim();
        
        if (!question) {
            alert('Please enter a question');
            return;
        }
        
        // Show loading indicators with better user feedback
        answerDiv.innerHTML = '<div class="loading-spinner"></div> Thinking...';
        sourcesDiv.innerHTML = '<div class="loading-text">Finding relevant information...</div>';
        askButton.disabled = true;
        
        try {
            const response = await fetch('/api/ask', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ query: question }),
            });
            
            if (!response.ok) {
                throw new Error(`Server error: ${response.status}`);
            }
            
            const data = await response.json();
            
            // Update answer
            answerDiv.innerHTML = data.answer || "No answer was provided.";
            
            // Update sources with better formatting
            if (data.sources && data.sources.length > 0) {
                let sourcesHTML = '<div class="sources-container">';
                data.sources.forEach((source, index) => {
                    sourcesHTML += `
                        <div class="source-item">
                            <h4>Source ${index + 1}</h4>
                            <p class="source-content">${source.content || "No content preview available"}</p>
                            <p class="source-metadata">
                                <span>Document: ${source.metadata?.source || "Unknown"}</span>
                                ${source.metadata?.page ? `<span>Page: ${source.metadata.page}</span>` : ''}
                            </p>
                        </div>
                    `;
                });
                sourcesHTML += '</div>';
                sourcesDiv.innerHTML = sourcesHTML;
            } else {
                sourcesDiv.innerHTML = '<p class="no-sources">No source documents were found for this question.</p>';
            }
            
        } catch (error) {
            console.error('Error:', error);
            answerDiv.innerHTML = `
                <div class="error-message">
                    Sorry, I encountered a problem while processing your question. 
                    <button class="retry-button">Try Again</button>
                </div>`;
            sourcesDiv.innerHTML = '';
            
            // Add retry functionality
            document.querySelector('.retry-button')?.addEventListener('click', () => {
                askQuestion();
            });
        } finally {
            askButton.disabled = false;
        }
    }
    
    // Allow pressing Enter to submit question
    questionInput.addEventListener('keypress', function(e) {
        if (e.key === 'Enter') {
            askButton.click();
        }
    });
});
