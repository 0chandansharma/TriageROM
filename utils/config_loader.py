#!/usr/bin/env python
# -*- coding: utf-8 -*-

import toml
import os
from typing import Dict, Any, Optional
from pathlib import Path

class ConfigLoader:
    """Configuration file loader and manager"""
    
    def __init__(self, config_dir: str = "config"):
        self.config_dir = Path(config_dir)
        self.loaded_configs = {}
    
    def load_config(self, config_name: str = "default_config") -> Dict[str, Any]:
        """
        Load configuration from TOML file
        
        Args:
            config_name: Name of config file (without .toml extension)
            
        Returns:
            Configuration dictionary
        """
        config_file = self.config_dir / f"{config_name}.toml"
        
        if config_name in self.loaded_configs:
            return self.loaded_configs[config_name].copy()
        
        try:
            if config_file.exists():
                with open(config_file, 'r') as f:
                    config_data = toml.load(f)
                self.loaded_configs[config_name] = config_data
                return config_data.copy()
            else:
                print(f"Warning: Config file {config_file} not found")
                default_config = {
                "api": {"host": "0.0.0.0", "port": 8000},
                "pose": {"model_type": "mediapipe"},
                "loweback": {"target_flexion_rom": 50.0, "target_extension_rom": 15.0}
                }
                return default_config   
                    
        except Exception as e:
            print(f"Error loading config {config_name}: {e}")
            return {}
    
    def merge_configs(self, *config_names: str) -> Dict[str, Any]:
        """
        Merge multiple configuration files
        Later configs override earlier ones
        
        Args:
            config_names: Names of config files to merge
            
        Returns:
            Merged configuration dictionary
        """
        merged_config = {}
        
        for config_name in config_names:
            config = self.load_config(config_name)
            merged_config = self._deep_merge(merged_config, config)
        
        return merged_config
    
    def _deep_merge(self, base_dict: Dict, override_dict: Dict) -> Dict:
        """Recursively merge dictionaries"""
        result = base_dict.copy()
        
        for key, value in override_dict.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._deep_merge(result[key], value)
            else:
                result[key] = value
        
        return result
    
    def get_config_value(self, config: Dict, key_path: str, default: Any = None) -> Any:
        """
        Get nested configuration value using dot notation
        
        Args:
            config: Configuration dictionary
            key_path: Dot-separated key path (e.g., 'api.host')
            default: Default value if key not found
            
        Returns:
            Configuration value or default
        """
        keys = key_path.split('.')
        current = config
        
        try:
            for key in keys:
                current = current[key]
            return current
        except (KeyError, TypeError):
            return default
    
    def validate_required_keys(self, config: Dict, required_keys: list) -> Dict:
        """
        Validate that required configuration keys are present
        
        Args:
            config: Configuration dictionary
            required_keys: List of required keys (supports dot notation)
            
        Returns:
            Validation results
        """
        validation_result = {
            "valid": True,
            "missing_keys": [],
            "errors": []
        }
        
        for key in required_keys:
            value = self.get_config_value(config, key)
            if value is None:
                validation_result["missing_keys"].append(key)
                validation_result["valid"] = False
        
        if validation_result["missing_keys"]:
            validation_result["errors"].append(
                f"Missing required configuration keys: {', '.join(validation_result['missing_keys'])}"
            )
        
        return validation_result