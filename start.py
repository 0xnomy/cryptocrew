import os
import sys
import time
import subprocess
from pathlib import Path

def check_env():
    if not Path(".env").exists():
        print("❌ .env file not found!")
        print("Please copy env.example to .env and add your GROQ_API_KEY")
        return False
    
    from dotenv import load_dotenv
    load_dotenv()
    
    if not os.getenv("GROQ_API_KEY"):
        print("❌ GROQ_API_KEY not found in .env file!")
        print("Please add your Groq API key to the .env file")
        return False
    
    print("✅ Environment check passed")
    return True

def start_chromadb():
    print("🚀 Starting ChromaDB...")
    try:
        # Start ChromaDB in a subprocess
        chroma_process = subprocess.Popen(
            ["chroma", "run", "--host", "0.0.0.0", "--port", "8000"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )
        
        # Wait for ChromaDB to start
        time.sleep(15)
        
        if chroma_process.poll() is None:
            print("✅ ChromaDB started successfully")
            return chroma_process
        else:
            print("❌ ChromaDB failed to start")
            return None
            
    except FileNotFoundError:
        print("❌ ChromaDB not found. Please install it with: pip install chromadb")
        return None
    except Exception as e:
        print(f"❌ Error starting ChromaDB: {e}")
        return None

def start_crewai_backend():
    print("🚀 Starting CrewAI Backend...")
    try:
        # Start CrewAI backend in a subprocess
        backend_process = subprocess.Popen(
            [sys.executable, "crewai_app.py"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )
        
        # Wait for backend to start
        time.sleep(5)
        
        if backend_process.poll() is None:
            print("✅ CrewAI Backend started successfully")
            return backend_process
        else:
            print("❌ CrewAI Backend failed to start")
            return None
            
    except Exception as e:
        print(f"❌ Error starting CrewAI Backend: {e}")
        return None

def start_frontend():
    print("🚀 Starting Frontend...")
    try:
        # Change to frontend directory and start Flask app
        frontend_process = subprocess.Popen(
            [sys.executable, "app.py"],
            cwd="frontend",
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )
        
        # Wait for frontend to start
        time.sleep(3)
        
        if frontend_process.poll() is None:
            print("✅ Frontend started successfully")
            return frontend_process
        else:
            print("❌ Frontend failed to start")
            return None
            
    except Exception as e:
        print(f"❌ Error starting Frontend: {e}")
        return None

def main():
    print("=" * 50)
    print("🚀 CryptoCrew Production Startup")
    print("=" * 50)
    
    # Check environment
    if not check_env():
        sys.exit(1)
    
    processes = []
    
    try:
        # Start ChromaDB
        chroma_process = start_chromadb()
        if chroma_process:
            processes.append(("ChromaDB", chroma_process))
        else:
            print("⚠️ ChromaDB failed to start, continuing without it...")
        
        # Start CrewAI Backend
        backend_process = start_crewai_backend()
        if backend_process:
            processes.append(("CrewAI Backend", backend_process))
        else:
            print("❌ Failed to start CrewAI Backend. Exiting.")
            sys.exit(1)
        
        # Start Frontend
        frontend_process = start_frontend()
        if frontend_process:
            processes.append(("Frontend", frontend_process))
        else:
            print("❌ Failed to start Frontend. Exiting.")
            sys.exit(1)
        
        print("\n" + "=" * 50)
        print("🎉 All services started successfully!")
        print("=" * 50)
        print("📱 Main App: http://localhost:3000")
        print("\nPress Ctrl+C to stop all services...")
        
        # Keep the script running
        while True:
            time.sleep(1)
            
    except KeyboardInterrupt:
        print("\n\n🛑 Stopping all services...")
        
        # Stop all processes
        for name, process in processes:
            print(f"Stopping {name}...")
            try:
                process.terminate()
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
            except Exception as e:
                print(f"Error stopping {name}: {e}")
        
        print("✅ All services stopped")
        sys.exit(0)
    
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        
        # Clean up processes
        for name, process in processes:
            try:
                process.terminate()
            except:
                pass
        
        sys.exit(1)

if __name__ == "__main__":
    main() 