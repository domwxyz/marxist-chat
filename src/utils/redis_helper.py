import json
import logging
import time
from typing import Dict, List, Any, Optional, Union
import redis
from redis.asyncio import Redis as AsyncRedis

import config

logger = logging.getLogger("utils.redis_helper")

class RedisHelper:
    """Helper class for Redis operations in distributed mode"""
    
    _client = None
    _async_client = None
    
    @classmethod
    def get_client(cls):
        """Get or create Redis client"""
        if cls._client is None:
            try:
                cls._client = redis.Redis(
                    host=config.REDIS_HOST,
                    port=config.REDIS_PORT,
                    decode_responses=True
                )
                # Test connection
                cls._client.ping()
                logger.info(f"Connected to Redis at {config.REDIS_HOST}:{config.REDIS_PORT}")
            except Exception as e:
                logger.error(f"Failed to connect to Redis: {e}")
                cls._client = None
                raise
        return cls._client
    
    @classmethod
    async def get_async_client(cls):
        """Get or create async Redis client"""
        if cls._async_client is None:
            try:
                cls._async_client = AsyncRedis(
                    host=config.REDIS_HOST,
                    port=config.REDIS_PORT,
                    decode_responses=True
                )
                # Test connection
                await cls._async_client.ping()
                logger.info(f"Connected to async Redis at {config.REDIS_HOST}:{config.REDIS_PORT}")
            except Exception as e:
                logger.error(f"Failed to connect to async Redis: {e}")
                cls._async_client = None
                raise
        return cls._async_client
    
    @classmethod
    def close(cls):
        """Close Redis connection"""
        if cls._client:
            cls._client.close()
            cls._client = None
    
    @classmethod
    async def close_async(cls):
        """Close async Redis connection"""
        if cls._async_client:
            await cls._async_client.close()
            cls._async_client = None
    
    @classmethod
    def set_dict(cls, key: str, data: Dict, expiry: int = None) -> bool:
        """Store dictionary data in Redis"""
        try:
            client = cls.get_client()
            serialized = json.dumps(data)
            result = client.set(key, serialized)
            if expiry:
                client.expire(key, expiry)
            return result
        except Exception as e:
            logger.error(f"Error in set_dict: {e}")
            return False
    
    @classmethod
    async def set_dict_async(cls, key: str, data: Dict, expiry: int = None) -> bool:
        """Store dictionary data in Redis asynchronously"""
        try:
            client = await cls.get_async_client()
            serialized = json.dumps(data)
            result = await client.set(key, serialized)
            if expiry:
                await client.expire(key, expiry)
            return result
        except Exception as e:
            logger.error(f"Error in set_dict_async: {e}")
            return False
    
    @classmethod
    def get_dict(cls, key: str) -> Optional[Dict]:
        """Retrieve dictionary data from Redis"""
        try:
            client = cls.get_client()
            data = client.get(key)
            if data:
                return json.loads(data)
            return None
        except Exception as e:
            logger.error(f"Error in get_dict: {e}")
            return None
    
    @classmethod
    async def get_dict_async(cls, key: str) -> Optional[Dict]:
        """Retrieve dictionary data from Redis asynchronously"""
        try:
            client = await cls.get_async_client()
            data = await client.get(key)
            if data:
                return json.loads(data)
            return None
        except Exception as e:
            logger.error(f"Error in get_dict_async: {e}")
            return None
    
    @classmethod
    def delete_key(cls, key: str) -> bool:
        """Delete a key from Redis"""
        try:
            client = cls.get_client()
            return client.delete(key) > 0
        except Exception as e:
            logger.error(f"Error in delete_key: {e}")
            return False
    
    @classmethod
    async def delete_key_async(cls, key: str) -> bool:
        """Delete a key from Redis asynchronously"""
        try:
            client = await cls.get_async_client()
            return await client.delete(key) > 0
        except Exception as e:
            logger.error(f"Error in delete_key_async: {e}")
            return False
    
    @classmethod
    def add_to_queue(cls, queue_name: str, item: Dict) -> bool:
        """Add an item to a Redis queue"""
        try:
            client = cls.get_client()
            serialized = json.dumps(item)
            return client.rpush(queue_name, serialized) > 0
        except Exception as e:
            logger.error(f"Error in add_to_queue: {e}")
            return False
    
    @classmethod
    async def add_to_queue_async(cls, queue_name: str, item: Dict) -> bool:
        """Add an item to a Redis queue asynchronously"""
        try:
            client = await cls.get_async_client()
            serialized = json.dumps(item)
            return await client.rpush(queue_name, serialized) > 0
        except Exception as e:
            logger.error(f"Error in add_to_queue_async: {e}")
            return False
    
    @classmethod
    def get_from_queue(cls, queue_name: str, block: bool = False, timeout: int = 0) -> Optional[Dict]:
        """Get an item from a Redis queue"""
        try:
            client = cls.get_client()
            if block:
                item = client.blpop(queue_name, timeout)
                if item:
                    return json.loads(item[1])
                return None
            else:
                item = client.lpop(queue_name)
                if item:
                    return json.loads(item)
                return None
        except Exception as e:
            logger.error(f"Error in get_from_queue: {e}")
            return None
    
    @classmethod
    async def get_from_queue_async(cls, queue_name: str, block: bool = False, timeout: int = 0) -> Optional[Dict]:
        """Get an item from a Redis queue asynchronously"""
        try:
            client = await cls.get_async_client()
            if block:
                item = await client.blpop(queue_name, timeout)
                if item:
                    return json.loads(item[1])
                return None
            else:
                item = await client.lpop(queue_name)
                if item:
                    return json.loads(item)
                return None
        except Exception as e:
            logger.error(f"Error in get_from_queue_async: {e}")
            return None
    
    @classmethod
    def get_queue_length(cls, queue_name: str) -> int:
        """Get the length of a Redis queue"""
        try:
            client = cls.get_client()
            return client.llen(queue_name)
        except Exception as e:
            logger.error(f"Error in get_queue_length: {e}")
            return 0
    
    @classmethod
    async def get_queue_length_async(cls, queue_name: str) -> int:
        """Get the length of a Redis queue asynchronously"""
        try:
            client = await cls.get_async_client()
            return await client.llen(queue_name)
        except Exception as e:
            logger.error(f"Error in get_queue_length_async: {e}")
            return 0
    
    @classmethod
    def register_instance(cls, instance_id: str, data: Dict) -> bool:
        """Register an API or LLM instance"""
        try:
            instance_key = f"instance:{instance_id}"
            data["last_seen"] = time.time()
            data["host"] = config.API_ID
            return cls.set_dict(instance_key, data, expiry=60)  # 60 second TTL
        except Exception as e:
            logger.error(f"Error in register_instance: {e}")
            return False
    
    @classmethod
    async def register_instance_async(cls, instance_id: str, data: Dict) -> bool:
        """Register an API or LLM instance asynchronously"""
        try:
            instance_key = f"instance:{instance_id}"
            data["last_seen"] = time.time()
            data["host"] = config.API_ID
            return await cls.set_dict_async(instance_key, data, expiry=60)  # 60 second TTL
        except Exception as e:
            logger.error(f"Error in register_instance_async: {e}")
            return False
    
    @classmethod
    def get_all_instances(cls, prefix: str = "instance:") -> List[Dict]:
        """Get all registered instances"""
        try:
            client = cls.get_client()
            keys = client.keys(f"{prefix}*")
            instances = []
            for key in keys:
                data = cls.get_dict(key)
                if data:
                    data["id"] = key.replace(prefix, "")
                    instances.append(data)
            return instances
        except Exception as e:
            logger.error(f"Error in get_all_instances: {e}")
            return []
    
    @classmethod
    async def get_all_instances_async(cls, prefix: str = "instance:") -> List[Dict]:
        """Get all registered instances asynchronously"""
        try:
            client = await cls.get_async_client()
            keys = await client.keys(f"{prefix}*")
            instances = []
            for key in keys:
                data = await cls.get_dict_async(key)
                if data:
                    data["id"] = key.replace(prefix, "")
                    instances.append(data)
            return instances
        except Exception as e:
            logger.error(f"Error in get_all_instances_async: {e}")
            return []
            