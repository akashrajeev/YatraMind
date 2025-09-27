import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { AlertTriangle, Home } from "lucide-react";
import { Link } from "react-router-dom";

const NotFound = () => {
  return (
    <div className="flex items-center justify-center min-h-[60vh]">
      <Card className="w-full max-w-md">
        <CardHeader className="text-center">
          <div className="mx-auto w-12 h-12 bg-destructive/10 rounded-full flex items-center justify-center mb-4">
            <AlertTriangle className="h-6 w-6 text-destructive" />
          </div>
          <CardTitle className="text-2xl">Page Not Found</CardTitle>
        </CardHeader>
        <CardContent className="text-center space-y-4">
          <p className="text-muted-foreground">
            The page you're looking for doesn't exist or has been moved.
          </p>
          <Button asChild>
            <Link to="/">
              <Home className="h-4 w-4 mr-2" />
              Return to Dashboard
            </Link>
          </Button>
        </CardContent>
      </Card>
    </div>
  );
};

export default NotFound;
