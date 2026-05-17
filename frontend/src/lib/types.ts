// Data types for users.json and catalog.json

export interface UserFeatures {
  days_since_login?: number;
  engagement_score?: number;
  avg_watch_time?: number;
  account_age_months?: number;
  monthly_fee?: number;
  devices_used?: number;
  watch_sessions_per_week?: number;
  binge_watch_sessions?: number;
  completion_rate?: number;
  rating_given?: number;
  content_interactions?: number;
  recommendation_click_rate?: number;
}

// Categorical fields that live at the top level of a User record
export type Gender = 'Male' | 'Female' | 'Other';
export type Country = 'USA' | 'UK' | 'Canada' | 'India' | 'Brazil' | 'France' | 'Germany' | 'Japan' | 'Spain';
export type SubscriptionType = 'Basic' | 'Standard' | 'Premium';
export type PaymentMethod = 'Credit Card' | 'Debit Card' | 'PayPal' | 'UPI';
export type PrimaryDevice = 'Laptop' | 'Mobile' | 'Tablet' | 'Smart TV';

export interface User {
  name: string;
  cluster: number;
  age: number;
  preferred_genre: string;
  gender?: Gender;
  country?: Country;
  subscription_type?: SubscriptionType;
  payment_method?: PaymentMethod;
  primary_device?: PrimaryDevice;
  features: UserFeatures;
}

export interface UserDatabase {
  [userId: string]: User;
}

export interface CatalogItem {
  title: string;
  type: string;
  genre: string;
  rating: string;
  duration: number;
  trending: number;
  ai_description: string;
  year: number;
  seasons?: number;
  popularity_score?: number;
  director?: string;
  cast?: string;
  country?: string;
}
