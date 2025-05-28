#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
from typing import Dict, List, Optional
import numpy as np

class MOTFileHandler:
    """
    Handle MOT (Motion) file generation and processing
    Adapted from Sports2D MOT file handling
    """
    
    def __init__(self):
        self.file_version = "1.0"
        
    def write_mot_file(self, file_path: str, motion_data: Dict, metadata: Optional[Dict] = None) -> bool:
        """
        Write motion data to MOT file format
        
        Args:
            file_path: Output file path
            motion_data: Dictionary with time and angle data
            metadata: Optional metadata for the file
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Extract data
            time_data = motion_data.get("time", [])
            angle_columns = {}
            
            # Collect all angle data columns
            for key, values in motion_data.items():
                if key != "time" and isinstance(values, (list, np.ndarray)):
                    angle_columns[key] = values
            
            if not time_data or not angle_columns:
                return False
            
            # Ensure all data has same length
            data_length = len(time_data)
            for column_name, column_data in angle_columns.items():
                if len(column_data) != data_length:
                    return False
            
            # Create header
            header_lines = self._create_mot_header(angle_columns, data_length, metadata)
            
            # Write file
            with open(file_path, 'w') as f:
                # Write header
                for line in header_lines:
                    f.write(line + '\n')
                
                # Write data
                for i in range(data_length):
                    # Time column
                    row_data = [f"{time_data[i]:.6f}"]
                    
                    # Angle columns
                    for column_name in sorted(angle_columns.keys()):
                        angle_value = angle_columns[column_name][i]
                        row_data.append(f"{angle_value:.6f}")
                    
                    f.write('\t'.join(row_data) + '\n')
            
            return True
            
        except Exception as e:
            print(f"Error writing MOT file: {e}")
            return False
    
    def _create_mot_header(self, angle_columns: Dict, data_length: int, metadata: Optional[Dict] = None) -> List[str]:
        """Create MOT file header"""
        
        column_names = ['time'] + sorted(angle_columns.keys())
        num_columns = len(column_names)
        
        # Default metadata
        if metadata is None:
            metadata = {}
        
        header_lines = [
            "Coordinates",
            f"version={self.file_version}",
            f"nRows={data_length}",
            f"nColumns={num_columns}",
            "inDegrees=yes",
            "",
            "Units are S.I. units (second, meters, Newtons, ...)",
            "If the header above contains a line with 'inDegrees', this indicates whether rotational values are in degrees (yes) or radians (no).",
            "",
            "endheader",
            '\t'.join(column_names)
        ]
        
        return header_lines
    
    def read_mot_file(self, file_path: str) -> Optional[Dict]:
        """
        Read MOT file and return motion data
        
        Args:
            file_path: MOT file path
            
        Returns:
            Dictionary with motion data or None if failed
        """
        try:
            if not os.path.exists(file_path):
                return None
            
            motion_data = {}
            header_complete = False
            column_names = []
            
            with open(file_path, 'r') as f:
                for line in f:
                    line = line.strip()
                    
                    if not header_complete:
                        if line == "endheader":
                            header_complete = True
                            continue
                        elif line.startswith(("nRows=", "nColumns=", "inDegrees=")):
                            # Parse header info if needed
                            continue
                    else:
                        # First line after header is column names
                        if not column_names:
                            column_names = line.split('\t')
                            for name in column_names:
                                motion_data[name] = []
                        else:
                            # Data lines
                            values = line.split('\t')
                            if len(values) == len(column_names):
                                for i, value in enumerate(values):
                                    try:
                                        motion_data[column_names[i]].append(float(value))
                                    except ValueError:
                                        motion_data[column_names[i]].append(0.0)
            
            return motion_data if motion_data else None
            
        except Exception as e:
            print(f"Error reading MOT file: {e}")
            return None
    
    def validate_mot_file(self, file_path: str) -> Dict:
        """
        Validate MOT file format and content
        
        Args:
            file_path: MOT file path
            
        Returns:
            Validation results
        """
        validation_result = {
            "valid": False,
            "errors": [],
            "warnings": [],
            "info": {}
        }
        
        try:
            if not os.path.exists(file_path):
                validation_result["errors"].append("File does not exist")
                return validation_result
            
            # Try to read the file
            motion_data = self.read_mot_file(file_path)
            
            if motion_data is None:
                validation_result["errors"].append("Could not parse MOT file")
                return validation_result
            
            # Check for required columns
            if "time" not in motion_data:
                validation_result["errors"].append("Missing 'time' column")
            
            # Check data consistency
            if motion_data:
                data_lengths = [len(values) for values in motion_data.values()]
                if len(set(data_lengths)) > 1:
                    validation_result["errors"].append("Inconsistent data lengths across columns")
                
                validation_result["info"]["num_columns"] = len(motion_data)
                validation_result["info"]["num_rows"] = data_lengths[0] if data_lengths else 0
                validation_result["info"]["columns"] = list(motion_data.keys())
            
            # If no errors, file is valid
            if not validation_result["errors"]:
                validation_result["valid"] = True
            
        except Exception as e:
            validation_result["errors"].append(f"Validation error: {str(e)}")
        
        return validation_result
    
    def get_mot_info(self, file_path: str) -> Optional[Dict]:
        """Get information about MOT file without loading all data"""
        try:
            info = {
                "file_path": file_path,
                "file_size": os.path.getsize(file_path),
                "columns": [],
                "num_rows": 0,
                "duration": 0.0
            }
            
            with open(file_path, 'r') as f:
                header_complete = False
                first_data_line = None
                last_data_line = None
                row_count = 0
                
                for line in f:
                    line = line.strip()
                    
                    if not header_complete:
                        if line == "endheader":
                            header_complete = True
                        elif line.startswith("nRows="):
                            try:
                                info["num_rows"] = int(line.split("=")[1])
                            except:
                                pass
                        continue
                    
                    # First line after header is column names
                    if not info["columns"]:
                        info["columns"] = line.split('\t')
                    else:
                        # Count data rows and get first/last time
                        row_count += 1
                        if first_data_line is None:
                            first_data_line = line
                        last_data_line = line
                
                # Calculate duration from first and last time values
                if first_data_line and last_data_line:
                    try:
                        first_time = float(first_data_line.split('\t')[0])
                        last_time = float(last_data_line.split('\t')[0])
                        info["duration"] = last_time - first_time
                    except:
                        pass
                
                info["actual_rows"] = row_count
            
            return info
            
        except Exception as e:
            print(f"Error getting MOT file info: {e}")
            return None