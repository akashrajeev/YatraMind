import os
from datetime import datetime

file_path = r'c:\Users\adith\OneDrive\Desktop\YatraMind\kmrl_train_induction_system\backend\app\api\trainsets.py'
append_content = '''

class ReviewCreate(BaseModel):
    rating: int = Field(..., ge=1, le=5, description="Rating from 1 to 5")
    comment: str = Field(..., min_length=1, max_length=500, description="Review comment")


@router.post("/{trainset_id}/review")
async def submit_review(
    trainset_id: str,
    review: ReviewCreate,
    current_user: User = Depends(require_role(UserRole.PASSENGER))
):
    """Submit a review for a trainset (Passenger only)"""
    if not trainset_id.replace("-", "").isalnum():
        raise HTTPException(status_code=400, detail="Invalid trainset_id format")
    try:
        collection = await cloud_db_manager.get_collection("trainset_reviews")
        
        review_doc = {
            "trainset_id": trainset_id,
            "user_id": current_user.id,
            "username": current_user.username,
            "rating": review.rating,
            "comment": review.comment,
            "created_at": datetime.now().isoformat()
        }
        
        await collection.insert_one(review_doc)
        
        logger.info(f"Review submitted for {trainset_id} by {current_user.username}")
        
        return {"message": "Review submitted successfully"}
        
    except Exception as e:
        logger.error(f"Error submitting review: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to submit review: {str(e)}")
'''

with open(file_path, 'a') as f:
    f.write(append_content)
print('Successfully appended content')
