import React, { useState, useEffect } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Star, Train } from "lucide-react";
import api from '../services/api';
import { toast } from "sonner";

interface Trainset {
    trainset_id: string;
    model: string;
    status: string;
}

const PassengerDashboard = () => {
    const [trainsets, setTrainsets] = useState<Trainset[]>([]);
    const [selectedTrain, setSelectedTrain] = useState<string | null>(null);
    const [rating, setRating] = useState(0);
    const [comment, setComment] = useState('');
    const [loading, setLoading] = useState(false);

    useEffect(() => {
        fetchTrainsets();
    }, []);

    const fetchTrainsets = async () => {
        try {
            const response = await api.get('/trainsets/');
            setTrainsets(response.data);
        } catch (error) {
            console.error('Error fetching trainsets:', error);
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
            await api.post(`/trainsets/${selectedTrain}/review`, {
                rating,
                comment
            });
            toast.success("Thank you for your feedback!");
            setRating(0);
            setComment('');
            setSelectedTrain(null);
        } catch (error) {
            console.error('Error submitting review:', error);
            toast.error("Failed to submit review");
        } finally {
            setLoading(false);
        }
    };

    return (
        <div className="max-w-4xl mx-auto p-6 space-y-8">
            <div className="text-center">
                <h1 className="text-3xl font-bold text-white mb-2">Passenger Feedback</h1>
                <p className="text-gray-400">Help us improve your journey</p>
            </div>

            <Card className="bg-gray-800 border-gray-700">
                <CardHeader>
                    <CardTitle className="text-white">Submit a Review</CardTitle>
                </CardHeader>
                <CardContent>
                    <form onSubmit={handleSubmitReview} className="space-y-6">
                        <div>
                            <label className="block text-sm font-medium text-gray-400 mb-2">
                                Select Train
                            </label>
                            <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                                {trainsets.map((train) => (
                                    <div
                                        key={train.trainset_id}
                                        onClick={() => setSelectedTrain(train.trainset_id)}
                                        className={`cursor-pointer p-4 rounded-lg border-2 transition-all ${selectedTrain === train.trainset_id
                                                ? "border-blue-500 bg-blue-900/20"
                                                : "border-gray-700 bg-gray-900/50 hover:border-gray-600"
                                            }`}
                                    >
                                        <Train className="h-6 w-6 text-gray-400 mb-2" />
                                        <div className="font-medium text-white">{train.trainset_id}</div>
                                        <div className="text-xs text-gray-500">{train.model}</div>
                                    </div>
                                ))}
                            </div>
                        </div>

                        <div>
                            <label className="block text-sm font-medium text-gray-400 mb-2">
                                Rating
                            </label>
                            <div className="flex space-x-2">
                                {[1, 2, 3, 4, 5].map((star) => (
                                    <button
                                        key={star}
                                        type="button"
                                        onClick={() => setRating(star)}
                                        className="focus:outline-none"
                                    >
                                        <Star
                                            className={`h-8 w-8 ${star <= rating ? "text-yellow-400 fill-yellow-400" : "text-gray-600"
                                                }`}
                                        />
                                    </button>
                                ))}
                            </div>
                        </div>

                        <div>
                            <label className="block text-sm font-medium text-gray-400 mb-2">
                                Comments
                            </label>
                            <textarea
                                value={comment}
                                onChange={(e) => setComment(e.target.value)}
                                className="w-full bg-gray-900 border border-gray-700 rounded-md p-3 text-white focus:outline-none focus:ring-2 focus:ring-blue-500"
                                rows={4}
                                placeholder="Tell us about your experience..."
                                required
                            />
                        </div>

                        <Button
                            type="submit"
                            disabled={loading}
                            className="w-full bg-blue-600 hover:bg-blue-700 text-white"
                        >
                            {loading ? "Submitting..." : "Submit Review"}
                        </Button>
                    </form>
                </CardContent>
            </Card>
        </div>
    );
};

export default PassengerDashboard;
