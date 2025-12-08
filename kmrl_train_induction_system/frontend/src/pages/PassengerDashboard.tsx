import React, { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Star, Train, LogOut, MessageSquare } from "lucide-react";
import api, { trainsetsApi } from '../services/api';
import { toast } from "sonner";
import { useAuth } from '../contexts/AuthContext';
import { ThemeToggle } from "@/components/theme-toggle";

interface Trainset {
    trainset_id: string;
    model: string;
    status: string;
}

interface Review {
    trainset_id: string;
    username: string;
    rating: number;
    comment: string;
    created_at: string;
}

const PassengerDashboard = () => {
    const [trainsets, setTrainsets] = useState<Trainset[]>([]);
    const [reviews, setReviews] = useState<Review[]>([]);
    const [selectedTrain, setSelectedTrain] = useState<string>("");
    const [rating, setRating] = useState(0);
    const [comment, setComment] = useState('');
    const [loading, setLoading] = useState(false);

    const { logout } = useAuth();
    const navigate = useNavigate();

    useEffect(() => {
        fetchTrainsets();
        fetchReviews();
    }, []);

    const fetchTrainsets = async () => {
        try {
            const response = await api.get('/v1/trainsets/');
            setTrainsets(response.data);
        } catch (error) {
            console.error('Error fetching trainsets:', error);
        }
    };

    const fetchReviews = async () => {
        try {
            const response = await trainsetsApi.getReviews();
            setReviews(response.data);
        } catch (error) {
            console.error('Error fetching reviews:', error);
        }
    };

    const handleLogout = async () => {
        try {
            await logout();
            navigate('/login');
            toast.success("Signed out successfully");
        } catch (error) {
            console.error('Logout error:', error);
            toast.error("Failed to sign out");
        }
    };

    const handleSubmitReview = async (e: React.FormEvent) => {
        e.preventDefault();
        if (!selectedTrain || rating === 0) {
            toast.error("Please select a train and rating");
            return;
        }

        setLoading(true);
        try {
            await api.post(`/v1/trainsets/${selectedTrain}/review`, {
                rating,
                comment
            });
            toast.success("Thank you for your feedback!");
            setRating(0);
            setComment('');
            setSelectedTrain("");
            fetchReviews(); // Refresh reviews after submission
        } catch (error) {
            console.error('Error submitting review:', error);
            toast.error("Failed to submit review");
        } finally {
            setLoading(false);
        }
    };

    return (
        <div className="min-h-screen bg-background p-6 transition-colors duration-300">
            <div className="max-w-4xl mx-auto space-y-8">
                <div className="flex justify-between items-center">
                    <div>
                        <h1 className="text-3xl font-bold text-foreground mb-2">Passenger Feedback</h1>
                        <p className="text-muted-foreground">Help us improve your journey</p>
                    </div>
                    <div className="flex items-center gap-4">
                        <ThemeToggle />
                        <Button
                            variant="destructive"
                            onClick={handleLogout}
                            className="flex items-center gap-2"
                        >
                            <LogOut className="h-4 w-4" />
                            Sign Out
                        </Button>
                    </div>
                </div>

                <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
                    {/* Submit Review Section */}
                    <Card className="bg-card border-border shadow-sm h-fit">
                        <CardHeader>
                            <CardTitle className="text-card-foreground">Submit a Review</CardTitle>
                        </CardHeader>
                        <CardContent>
                            <form onSubmit={handleSubmitReview} className="space-y-6">
                                <div>
                                    <label className="block text-sm font-medium text-muted-foreground mb-2">
                                        Select Train
                                    </label>
                                    <div className="relative">
                                        <Train className="absolute left-3 top-1/2 transform -translate-y-1/2 h-5 w-5 text-muted-foreground" />
                                        <select
                                            value={selectedTrain}
                                            onChange={(e) => setSelectedTrain(e.target.value)}
                                            className="w-full pl-10 pr-4 py-2 bg-background border border-input rounded-md text-foreground focus:outline-none focus:ring-2 focus:ring-ring focus:border-input appearance-none"
                                        >
                                            <option value="" disabled>Select a train to review...</option>
                                            {trainsets.map((train) => (
                                                <option key={train.trainset_id} value={train.trainset_id}>
                                                    {train.trainset_id} - {train.model}
                                                </option>
                                            ))}
                                        </select>
                                    </div>
                                </div>

                                <div>
                                    <label className="block text-sm font-medium text-muted-foreground mb-2">
                                        Rating
                                    </label>
                                    <div className="flex space-x-2">
                                        {[1, 2, 3, 4, 5].map((star) => (
                                            <button
                                                key={star}
                                                type="button"
                                                onClick={() => setRating(star)}
                                                className="focus:outline-none transition-transform hover:scale-110"
                                            >
                                                <Star
                                                    className={`h-8 w-8 ${star <= rating ? "text-yellow-400 fill-yellow-400" : "text-muted-foreground/30"
                                                        }`}
                                                />
                                            </button>
                                        ))}
                                    </div>
                                </div>

                                <div>
                                    <label className="block text-sm font-medium text-muted-foreground mb-2">
                                        Comments
                                    </label>
                                    <textarea
                                        value={comment}
                                        onChange={(e) => setComment(e.target.value)}
                                        className="w-full bg-background border border-input rounded-md p-3 text-foreground focus:outline-none focus:ring-2 focus:ring-ring placeholder:text-muted-foreground"
                                        rows={4}
                                        placeholder="Tell us about your experience..."
                                        required
                                    />
                                </div>

                                <Button
                                    type="submit"
                                    disabled={loading}
                                    className="w-full"
                                >
                                    {loading ? "Submitting..." : "Submit Review"}
                                </Button>
                            </form>
                        </CardContent>
                    </Card>

                    {/* Recent Reviews Section */}
                    <Card className="bg-card border-border shadow-sm h-fit">
                        <CardHeader>
                            <CardTitle className="text-card-foreground flex items-center gap-2">
                                <MessageSquare className="h-5 w-5" />
                                Recent Reviews
                            </CardTitle>
                        </CardHeader>
                        <CardContent>
                            <div className="space-y-4 max-h-[600px] overflow-y-auto pr-2">
                                {reviews.length === 0 ? (
                                    <div className="text-center text-muted-foreground py-8">
                                        No reviews yet. Be the first to share your experience!
                                    </div>
                                ) : (
                                    reviews.map((review, index) => (
                                        <div key={index} className="bg-muted/30 p-4 rounded-lg border border-border">
                                            <div className="flex justify-between items-start mb-2">
                                                <div>
                                                    <div className="font-medium text-foreground">{review.trainset_id}</div>
                                                    <div className="text-xs text-muted-foreground">
                                                        by {review.username} • {new Date(review.created_at).toLocaleDateString()}
                                                    </div>
                                                </div>
                                                <div className="flex">
                                                    {[...Array(5)].map((_, i) => (
                                                        <Star
                                                            key={i}
                                                            className={`h-3 w-3 ${i < review.rating ? "text-yellow-400 fill-yellow-400" : "text-muted-foreground/30"}`}
                                                        />
                                                    ))}
                                                </div>
                                            </div>
                                            <p className="text-sm text-foreground/90">{review.comment}</p>
                                        </div>
                                    ))
                                )}
                            </div>
                        </CardContent>
                    </Card>
                </div>
            </div>
        </div>
    );
};

export default PassengerDashboard;
