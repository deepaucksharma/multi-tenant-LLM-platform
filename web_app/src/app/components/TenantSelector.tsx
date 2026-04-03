'use client';

interface TenantSelectorProps {
  currentTenant: string;
  onTenantChange: (tenant: string) => void;
}

export default function TenantSelector({ currentTenant, onTenantChange }: TenantSelectorProps) {
  return (
    <select
      value={currentTenant}
      onChange={(e) => onTenantChange(e.target.value)}
      className="bg-gray-100 border border-gray-300 rounded-md px-3 py-1.5 text-sm
                 focus:outline-none focus:ring-2 focus:ring-blue-500"
    >
      <option value="sis">SIS — Education</option>
      <option value="mfg">MFG — Manufacturing</option>
    </select>
  );
}
