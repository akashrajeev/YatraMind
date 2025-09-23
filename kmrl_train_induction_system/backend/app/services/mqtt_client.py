# backend/app/services/mqtt_client.py
import asyncio
import json
import logging
from datetime import datetime
from typing import Dict, Any, Callable, Optional
import paho.mqtt.client as mqtt
from app.config import settings

logger = logging.getLogger(__name__)

class MQTTClient:
    """MQTT client for real-time IoT sensor data streaming"""
    
    def __init__(self):
        self.client = None
        self.connected = False
        self.subscriptions = {}
        self.message_handlers = {}
        
    async def connect(self):
        """Connect to MQTT broker"""
        try:
            self.client = mqtt.Client()
            self.client.on_connect = self._on_connect
            self.client.on_message = self._on_message
            self.client.on_disconnect = self._on_disconnect
            
            # Configure TLS if needed
            if getattr(settings, 'mqtt_use_tls', None) == 'true':
                self.client.tls_set()
            
            # Connect to broker
            broker_host = getattr(settings, 'mqtt_broker_host', settings.mqtt_broker)
            broker_port = int(getattr(settings, 'mqtt_broker_port', settings.mqtt_port))
            
            self.client.connect(broker_host, broker_port, 60)
            self.client.loop_start()
            
            logger.info(f"MQTT client connecting to {broker_host}:{broker_port}")
            
        except Exception as e:
            logger.error(f"MQTT connection failed: {e}")
            raise
    
    async def disconnect(self):
        """Disconnect from MQTT broker"""
        try:
            if self.client:
                self.client.loop_stop()
                self.client.disconnect()
                self.connected = False
                logger.info("MQTT client disconnected")
        except Exception as e:
            logger.error(f"MQTT disconnect error: {e}")
    
    def _on_connect(self, client, userdata, flags, rc):
        """Callback for MQTT connection"""
        if rc == 0:
            self.connected = True
            logger.info("MQTT client connected successfully")
            
            # Resubscribe to all topics
            for topic in self.subscriptions.keys():
                client.subscribe(topic)
                logger.info(f"Resubscribed to topic: {topic}")
        else:
            logger.error(f"MQTT connection failed with code {rc}")
    
    def _on_message(self, client, userdata, msg):
        """Callback for MQTT messages"""
        try:
            topic = msg.topic
            payload = msg.payload.decode('utf-8')
            
            logger.debug(f"Received MQTT message on {topic}: {payload}")
            
            # Parse JSON payload
            try:
                data = json.loads(payload)
            except json.JSONDecodeError:
                logger.warning(f"Invalid JSON in MQTT message: {payload}")
                return
            
            # Route message to appropriate handler
            if topic in self.message_handlers:
                handler = self.message_handlers[topic]
                asyncio.create_task(handler(data))
            else:
                logger.warning(f"No handler for topic: {topic}")
                
        except Exception as e:
            logger.error(f"Error processing MQTT message: {e}")
    
    def _on_disconnect(self, client, userdata, rc):
        """Callback for MQTT disconnection"""
        self.connected = False
        logger.warning(f"MQTT client disconnected with code {rc}")
    
    async def subscribe(self, topic: str, handler: Callable[[Dict[str, Any]], None]):
        """Subscribe to MQTT topic with message handler"""
        try:
            if not self.connected:
                await self.connect()
            
            self.client.subscribe(topic)
            self.subscriptions[topic] = True
            self.message_handlers[topic] = handler
            
            logger.info(f"Subscribed to topic: {topic}")
            
        except Exception as e:
            logger.error(f"Failed to subscribe to {topic}: {e}")
            raise
    
    async def unsubscribe(self, topic: str):
        """Unsubscribe from MQTT topic"""
        try:
            if self.client and self.connected:
                self.client.unsubscribe(topic)
                self.subscriptions.pop(topic, None)
                self.message_handlers.pop(topic, None)
                logger.info(f"Unsubscribed from topic: {topic}")
        except Exception as e:
            logger.error(f"Failed to unsubscribe from {topic}: {e}")
    
    async def publish(self, topic: str, message: Dict[str, Any]):
        """Publish message to MQTT topic"""
        try:
            if not self.connected:
                await self.connect()
            
            payload = json.dumps(message)
            result = self.client.publish(topic, payload)
            
            if result.rc == mqtt.MQTT_ERR_SUCCESS:
                logger.debug(f"Published message to {topic}")
            else:
                logger.error(f"Failed to publish to {topic}: {result.rc}")
                
        except Exception as e:
            logger.error(f"Failed to publish to {topic}: {e}")
            raise

class IoTDataStreamer:
    """IoT data streaming service using MQTT"""
    
    def __init__(self):
        self.mqtt_client = MQTTClient()
        self.sensor_topics = {
            "bogie_monitoring": "kmrl/sensors/bogie",
            "brake_system": "kmrl/sensors/brake",
            "hvac_control": "kmrl/sensors/hvac",
            "door_mechanism": "kmrl/sensors/door",
            "pantograph": "kmrl/sensors/pantograph"
        }
    
    async def start_streaming(self):
        """Start IoT data streaming"""
        try:
            await self.mqtt_client.connect()
            
            # Subscribe to sensor topics
            for sensor_type, topic in self.sensor_topics.items():
                await self.mqtt_client.subscribe(topic, self._handle_sensor_data)
            
            logger.info("IoT data streaming started")
            
        except Exception as e:
            logger.error(f"Failed to start IoT streaming: {e}")
            raise
    
    async def stop_streaming(self):
        """Stop IoT data streaming"""
        try:
            await self.mqtt_client.disconnect()
            logger.info("IoT data streaming stopped")
        except Exception as e:
            logger.error(f"Failed to stop IoT streaming: {e}")
    
    async def _handle_sensor_data(self, data: Dict[str, Any]):
        """Handle incoming sensor data"""
        try:
            # Process sensor data
            sensor_type = data.get("sensor_type", "unknown")
            trainset_id = data.get("trainset_id", "unknown")
            
            logger.info(f"Processing {sensor_type} data for {trainset_id}")
            
            # Store in InfluxDB (via cloud_db_manager)
            from app.utils.cloud_database import cloud_db_manager
            await cloud_db_manager.write_sensor_data(data)
            
        except Exception as e:
            logger.error(f"Error handling sensor data: {e}")

# Global MQTT client instance
mqtt_client = MQTTClient()
iot_streamer = IoTDataStreamer()
