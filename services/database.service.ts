import { Pool } from 'pg';
import { pool } from '../config/database.config';
import { AddressMatcher } from '../services/ml.service';

export async function findAddressMatches(address: string, threshold: number = 0.8): Promise<any[]> {
    // Get all addresses from database for ML comparison
    const query = `
        SELECT 
            a.raw_address,
            a.normalized_address,
            a.components,
            a.metadata->>'member_id' as member_id,
            a.metadata->>'member_name' as member_name,
            a.metadata->'original_record'->>'LIC' as lic,
            similarity(a.normalized_address, $1) as match_score
        FROM addresses a
        WHERE similarity(a.normalized_address, $1) > $2
        ORDER BY match_score DESC
        LIMIT 5;
    `;

    const result = await pool.query(query, [address, threshold]);
    return result.rows;
}