sio = None
socket_manager = None
app.add_middleware(GZipMiddleware, minimum_size=1024)

cors_origins = [origin.strip() for origin in settings.cors_origins.split(",") if origin.strip()]
app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"],
    allow_headers=["Authorization", "Content-Type", "X-API-Key"],
)

if _HAS_PROM:
    Instrumentator().instrument(app).expose(app, include_in_schema=False)

app.include_router(auth.router, prefix="/api/v1/auth", tags=["Authentication"])
app.include_router(users.router, prefix="/api/v1/users", tags=["Users"])
app.include_router(trainsets.router, prefix="/api/v1/trainsets", tags=["Trainsets"])
app.include_router(optimization.router, prefix="/api/v1/optimization", tags=["Optimization"])
app.include_router(optimization.router, prefix="/api/optimization", tags=["Optimization Legacy"])
app.include_router(dashboard.router, prefix="/api/v1/dashboard", tags=["Dashboard"])
app.include_router(ingestion.router, prefix="/api/v1/ingestion", tags=["Ingestion"])
app.include_router(assignments.router, prefix="/api/v1/assignments", tags=["Assignments"])
app.include_router(reports.router, prefix="/api/v1/reports", tags=["Reports"])
app.include_router(simulation.router, prefix="/api/v1/simulation", tags=["Simulation"])
app.include_router(notifications.router, prefix="/api/v1/notifications", tags=["Notifications"])
app.include_router(multi_depot_simulation.router, prefix="/api/v1/multi-depot", tags=["Multi-Depot Simulation"])

if _HAS_SOCKETIO and sio:
    @sio.event
    async def connect(sid, environ):
        logger.info("Client connected: %s", sid)
        await sio.emit("status", {"message": "Connected to KMRL Operations Dashboard"}, room=sid)

    @sio.event
    async def disconnect(sid):
        logger.info("Client disconnected: %s", sid)

    @sio.event
    async def join_room(sid, data):
        room = data.get("room", "general")
        sio.enter_room(sid, room)
        await sio.emit("joined_room", {"room": room}, room=sid)

    @sio.event
    async def leave_room(sid, data):
        room = data.get("room", "general")
        sio.leave_room(sid, room)
        await sio.emit("left_room", {"room": room}, room=sid)
