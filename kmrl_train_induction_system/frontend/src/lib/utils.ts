import { type ClassValue, clsx } from "clsx";
import { twMerge } from "tailwind-merge";

export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs));
}

export function formatDate(date: Date | string) {
  return new Intl.DateTimeFormat("en-US", {
    month: "short",
    day: "numeric",
    year: "numeric",
  }).format(new Date(date));
}

export function formatTime(date: Date | string) {
  return new Intl.DateTimeFormat("en-US", {
    hour: "2-digit",
    minute: "2-digit",
  }).format(new Date(date));
}

export function formatDateTime(date: Date | string) {
  return new Intl.DateTimeFormat("en-US", {
    month: "short",
    day: "numeric",
    year: "numeric",
    hour: "2-digit",
    minute: "2-digit",
  }).format(new Date(date));
}

export function getStatusColor(status: string) {
  switch (status.toLowerCase()) {
    case "active":
    case "online":
    case "completed":
    case "approved":
      return "text-success";
    case "pending":
    case "warning":
    case "in-progress":
      return "text-warning";
    case "error":
    case "failed":
    case "rejected":
      return "text-destructive";
    default:
      return "text-muted-foreground";
  }
}

export function getStatusBadgeVariant(status: string) {
  switch (status.toLowerCase()) {
    case "active":
    case "online":
    case "completed":
    case "approved":
      return "default";
    case "pending":
    case "warning":
    case "in-progress":
      return "secondary";
    case "error":
    case "failed":
    case "rejected":
      return "destructive";
    default:
      return "outline";
  }
}

export function generateId() {
  return Math.random().toString(36).substr(2, 9);
}

export function debounce<T extends (...args: any[]) => any>(
  func: T,
  wait: number
): (...args: Parameters<T>) => void {
  let timeout: NodeJS.Timeout;
  return (...args: Parameters<T>) => {
    clearTimeout(timeout);
    timeout = setTimeout(() => func(...args), wait);
  };
}

export function throttle<T extends (...args: any[]) => any>(
  func: T,
  limit: number
): (...args: Parameters<T>) => void {
  let inThrottle: boolean;
  return (...args: Parameters<T>) => {
    if (!inThrottle) {
      func(...args);
      inThrottle = true;
      setTimeout(() => (inThrottle = false), limit);
    }
  };
}
