import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Upload, X, File } from "lucide-react";
import { useState } from "react";

interface FileUploadProps {
  onFileSelect: (file: File | File[]) => void;
  accept?: string;
  maxSize?: number; // in MB
  disabled?: boolean;
  className?: string;
  multiple?: boolean;
}

export function FileUpload({
  onFileSelect,
  accept = "*/*",
  maxSize = 10,
  disabled = false,
  className = "",
  multiple = false
}: FileUploadProps) {
  const [selectedFiles, setSelectedFiles] = useState<File[]>([]);
  const [dragActive, setDragActive] = useState(false);

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const files = Array.from(e.target.files || []);
    if (files.length > 0) {
      if (files.some(file => file.size > maxSize * 1024 * 1024)) {
        alert(`Each file must be less than ${maxSize}MB`);
        return;
      }

      if (multiple) {
        setSelectedFiles(files);
        onFileSelect(files);
      } else {
        setSelectedFiles([files[0]]);
        onFileSelect(files[0]);
      }
    }
  };

  const handleDrag = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.type === "dragenter" || e.type === "dragover") {
      setDragActive(true);
    } else if (e.type === "dragleave") {
      setDragActive(false);
    }
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(false);

    const files = Array.from(e.dataTransfer.files || []);
    if (files.length > 0) {
      if (files.some(file => file.size > maxSize * 1024 * 1024)) {
        alert(`Each file must be less than ${maxSize}MB`);
        return;
      }

      if (multiple) {
        setSelectedFiles(files);
        onFileSelect(files);
      } else {
        setSelectedFiles([files[0]]);
        onFileSelect(files[0]);
      }
    }
  };

  const removeFile = (index: number) => {
    const newFiles = [...selectedFiles];
    newFiles.splice(index, 1);
    setSelectedFiles(newFiles);

    if (multiple) {
      onFileSelect(newFiles);
    } else {
      onFileSelect(newFiles[0] || null);
    }
  };

  return (
    <div className={`space-y-2 ${className}`}>
      <Label>File Upload</Label>
      <div
        className={`relative border-2 border-dashed rounded-lg p-6 transition-colors ${dragActive
            ? "border-primary bg-primary/5"
            : "border-muted-foreground/25 hover:border-muted-foreground/50"
          } ${disabled ? "opacity-50 cursor-not-allowed" : "cursor-pointer"}`}
        onDragEnter={handleDrag}
        onDragLeave={handleDrag}
        onDragOver={handleDrag}
        onDrop={handleDrop}
      >
        <Input
          type="file"
          accept={accept}
          onChange={handleFileChange}
          disabled={disabled}
          multiple={multiple}
          className="absolute inset-0 w-full h-full opacity-0 cursor-pointer"
        />

        {selectedFiles.length > 0 ? (
          <div className="space-y-2">
            {selectedFiles.map((file, index) => (
              <div key={index} className="flex items-center justify-between bg-muted/50 p-2 rounded">
                <div className="flex items-center gap-2 overflow-hidden">
                  <File className="h-4 w-4 text-muted-foreground flex-shrink-0" />
                  <span className="text-sm font-medium truncate">{file.name}</span>
                  <span className="text-xs text-muted-foreground flex-shrink-0">
                    ({(file.size / 1024 / 1024).toFixed(2)} MB)
                  </span>
                </div>
                <Button
                  type="button"
                  variant="ghost"
                  size="sm"
                  onClick={(e) => {
                    e.preventDefault();
                    removeFile(index);
                  }}
                  className="h-6 w-6 p-0"
                >
                  <X className="h-3 w-3" />
                </Button>
              </div>
            ))}
          </div>
        ) : (
          <div className="text-center">
            <Upload className="h-8 w-8 text-muted-foreground mx-auto mb-2" />
            <p className="text-sm text-muted-foreground">
              Drag and drop {multiple ? 'files' : 'a file'} here, or click to select
            </p>
            <p className="text-xs text-muted-foreground mt-1">
              Max size: {maxSize}MB
            </p>
          </div>
        )}
      </div>
    </div>
  );
}
