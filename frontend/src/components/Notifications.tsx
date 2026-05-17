'use client';
import { useUIStore } from '@/store';

export function Notifications() {
  const notifications = useUIStore((s) => s.notifications);
  return (
    <div className="fixed top-5 right-5 z-[1000] flex flex-col gap-2">
      {notifications.map((n) => (
        <div key={n.id} className="notification">
          <span>{n.message}</span>
        </div>
      ))}
    </div>
  );
}
