

from flask import Flask, jsonify, request, render_template_string
from .sokoban import SokobanEnv
from ragen.utils import set_seed

app = Flask(__name__)
env = SokobanEnv()

HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Sokoban Game</title>
    <style>
        body {
            font-family: monospace;
            display: flex;
            flex-direction: column;
            align-items: center;
            margin: 0;
            padding: 0;
        }
        #game-container {
            margin-top: 20px;
        }
        pre {
            background: #f4f4f4;
            padding: 10px;
            border: 1px solid #ccc;
            white-space: pre-wrap;
        }
    </style>
</head>
<body>
    <h1>Sokoban Game</h1>
    <div id="game-container">
        <pre id="game-board">Loading...</pre>
    </div>

    <script>
        const actions = {
            ArrowUp: 1,
            ArrowDown: 2,
            ArrowLeft: 3,
            ArrowRight: 4,
        };

        async function resetGame() {
            const response = await fetch('/reset');
            const data = await response.json();
            updateBoard(data.grid);
        }

        async function stepGame(action) {
            const response = await fetch('/step', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ action }),
            });
            const data = await response.json();
            if (data.done) {
                alert('Game over! Resetting...');
                await resetGame();
            } else {
                updateBoard(data.grid);
            }
        }

        function updateBoard(grid) {
            document.getElementById('game-board').innerText = grid;
        }

        document.addEventListener('keydown', async (event) => {
            if (actions[event.key] !== undefined) {
                await stepGame(actions[event.key]);
            }
        });

        resetGame();
    </script>
</body>
</html>
"""

@app.route('/')
def index():
    return render_template_string(HTML_TEMPLATE)

@app.route('/reset', methods=['GET'])
def reset():
    # with set_seed(2):
    env.reset()
    grid = env.render()
    return jsonify({'grid': grid})

@app.route('/step', methods=['POST'])
def step():
    action = request.json.get('action', 0)
    _, reward, done, _ = env.step(action)
    grid = env.render()
    return jsonify({'grid': grid, 'reward': reward, 'done': done})

@app.route('/close', methods=['GET'])
def close():
    env.close()
    return jsonify({'message': 'Environment closed'})

if __name__ == '__main__':
    app.run(debug=True)