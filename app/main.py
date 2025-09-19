from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.database.migrations import run_migrations
from app.controllers import auth, users, issues, reports

# Run migrations on startup
#run_migrations()

app = FastAPI(title="Citizen Issue Reporting System")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API routers
app.include_router(auth.router, prefix="/api/auth", tags=["authentication"])
app.include_router(users.router, prefix="/api/users", tags=["users"])
app.include_router(issues.router, prefix="/api/issues", tags=["issues"])
app.include_router(reports.router, prefix="/api/reports", tags=["reports"])

@app.get("/")
async def root():
    return {"message": "Welcome to the Citizen Issue Reporting System API"}

@app.get("/health")
async def health_check():
    return {"status": "healthy"}