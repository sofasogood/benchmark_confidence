import http.server
import socketserver
import webbrowser
import os
from pathlib import Path

def serve_visualizations():
    # Get the absolute path to the metrics_results directory
    metrics_dir = Path("data/metrics_results").absolute()
    
    # Change to the metrics directory
    os.chdir(metrics_dir)
    
    # Create a simple HTTP server
    PORT = 8000
    Handler = http.server.SimpleHTTPRequestHandler
    
    # Open browser tabs for both visualizations
    webbrowser.open(f'http://localhost:{PORT}/confidence_table.html')
    webbrowser.open(f'http://localhost:{PORT}/confidence_radar.html')
    
    # Start the server
    with socketserver.TCPServer(("", PORT), Handler) as httpd:
        print(f"Serving visualizations at http://localhost:{PORT}")
        print("Press Ctrl+C to stop the server")
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nShutting down server...")
            httpd.server_close()

if __name__ == "__main__":
    serve_visualizations() 